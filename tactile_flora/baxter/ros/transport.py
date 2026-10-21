"""The thin pub/sub interface the Baxter stack uses instead of rospy.

Owns: the ``Transport`` Protocol (publish, subscribe, advertise, call, now, sleep), the
in-memory ``FakeTransport`` used by tests, and ``RosTransport`` (ROS 1; rospy is imported lazily
inside it because it is not installable from PyPI).
Must never: hold node or recorder logic, or import rospy at module import time.

Messages are plain dicts. Topic payloads: tactile raw/force ``{"data": [4 floats]}``; health
``{"variance": [4], "saturation": [4], "baseline_drift": [4]}``; cameras ``{"image": HxWx3 uint8}``;
gripper state ``{"position": closure 0 open..1 closed, "command": float}``; policy status a dict of
parameters (see recorder.py). RosTransport codecs are UNVERIFIED (no ROS on the dev machine).
"""

from __future__ import annotations

import importlib
import json
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

Msg = dict[str, Any]
Callback = Callable[[Msg, float], None]
Handler = Callable[[Msg], Msg]


class Transport(Protocol):
    """Publish / subscribe / request-reply plus a clock."""

    def publish(self, topic: str, msg: Msg, stamp: float) -> None:
        """Publish ``msg`` stamped with ``stamp`` seconds."""
        ...

    def subscribe(self, topic: str, cb: Callback) -> None:
        """Call ``cb(msg, stamp)`` for every message on ``topic``."""
        ...

    def advertise(self, service: str, handler: Handler) -> None:
        """Serve ``service`` with ``handler(request) -> reply``."""
        ...

    def call(self, service: str, request: Msg) -> Msg:
        """Call a service and return its reply."""
        ...

    def now(self) -> float:
        """Return the current time in seconds on the transport clock."""
        ...

    def sleep(self, seconds: float) -> None:
        """Sleep (or advance the fake clock)."""
        ...


class FakeTransport:
    """In-memory transport with a manual clock; delivers synchronously."""

    def __init__(self, t0: float = 0.0) -> None:
        self._now = t0
        self.published: list[tuple[str, Msg, float]] = []
        self._subs: dict[str, list[Callback]] = defaultdict(list)
        self._services: dict[str, Handler] = {}

    def publish(self, topic: str, msg: Msg, stamp: float) -> None:
        """Record the message and deliver it to subscribers."""
        self.published.append((topic, msg, stamp))
        for cb in list(self._subs[topic]):
            cb(msg, stamp)

    def subscribe(self, topic: str, cb: Callback) -> None:
        """Register a subscriber."""
        self._subs[topic].append(cb)

    def advertise(self, service: str, handler: Handler) -> None:
        """Register a service handler."""
        self._services[service] = handler

    def call(self, service: str, request: Msg) -> Msg:
        """Invoke a registered service."""
        if service not in self._services:
            raise RuntimeError(f"service {service} is not advertised")
        return self._services[service](request)

    def now(self) -> float:
        """Manual clock."""
        return self._now

    def sleep(self, seconds: float) -> None:
        """Advance the manual clock."""
        self._now += seconds

    def messages(self, topic: str) -> list[tuple[Msg, float]]:
        """Return the published (msg, stamp) pairs of one topic."""
        return [(m, s) for t, m, s in self.published if t == topic]


class RosTransport:
    """ROS 1 transport. Topic types are chosen by a small codec table (UNVERIFIED on hardware)."""

    def __init__(self, node_name: str = "tactile_flora") -> None:
        try:
            self._rospy = importlib.import_module("rospy")
            self._std = importlib.import_module("std_msgs.msg")
            self._sensor = importlib.import_module("sensor_msgs.msg")
            self._baxter = importlib.import_module("baxter_core_msgs.msg")
        except ImportError as exc:
            raise RuntimeError("ROS 1 (rospy) is not available on this machine") from exc
        self._rospy.init_node(node_name, anonymous=False)
        self._pubs: dict[str, Any] = {}

    def _json(self, msg: Msg) -> Any:
        return self._std.String(data=json.dumps(msg))

    def publish(self, topic: str, msg: Msg, stamp: float) -> None:
        """Publish: Float32MultiArray for {"data"}, JSON String otherwise."""
        if topic not in self._pubs:
            typ = self._std.Float32MultiArray if set(msg) == {"data"} else self._std.String
            self._pubs[topic] = self._rospy.Publisher(topic, typ, queue_size=1000)
        if set(msg) == {"data"}:
            self._pubs[topic].publish(self._std.Float32MultiArray(data=msg["data"]))
        else:
            self._pubs[topic].publish(self._json(msg))

    def subscribe(self, topic: str, cb: Callback) -> None:
        """Subscribe; images and gripper states are decoded, anything else is JSON/array."""
        rospy = self._rospy

        def _stamp(m: Any) -> float:
            h = getattr(m, "header", None)
            if h is not None and h.stamp.to_sec() > 0:
                return float(h.stamp.to_sec())
            return float(rospy.get_time())

        def _on_image(m: Any) -> None:
            img = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, -1)
            if m.encoding in ("bgr8", "bgra8"):
                img = img[:, :, 2::-1]
            cb({"image": np.ascontiguousarray(img[:, :, :3])}, _stamp(m))

        def _on_gripper(m: Any) -> None:
            cb(
                {"position": 1.0 - float(m.position) / 100.0, "command": float(m.position)},
                _stamp(m),
            )

        def _on_array(m: Any) -> None:
            cb({"data": list(m.data)}, float(rospy.get_time()))

        def _on_json(m: Any) -> None:
            cb(json.loads(m.data), float(rospy.get_time()))

        if "image" in topic:
            rospy.Subscriber(topic, self._sensor.Image, _on_image, queue_size=60)
        elif "gripper" in topic:
            rospy.Subscriber(topic, self._baxter.EndEffectorState, _on_gripper, queue_size=200)
        elif topic.endswith(("/raw", "/force")):
            rospy.Subscriber(topic, self._std.Float32MultiArray, _on_array, queue_size=2000)
        else:
            rospy.Subscriber(topic, self._std.String, _on_json, queue_size=100)

    def advertise(self, service: str, handler: Handler) -> None:
        """Serve a JSON-over-String request/reply service via rospy.Service (std_srvs-free)."""
        srv = importlib.import_module("std_srvs.srv")

        def _h(_req: Any) -> Any:
            return srv.TriggerResponse(success=True, message=json.dumps(handler({})))

        self._rospy.Service(service, srv.Trigger, _h)

    def call(self, service: str, request: Msg) -> Msg:
        """Call a Trigger service whose message is the JSON reply."""
        srv = importlib.import_module("std_srvs.srv")
        self._rospy.wait_for_service(service, timeout=5.0)
        reply = self._rospy.ServiceProxy(service, srv.Trigger)()
        return json.loads(reply.message)  # type: ignore[no-any-return]

    def now(self) -> float:
        """ROS time."""
        return float(self._rospy.get_time())

    def sleep(self, seconds: float) -> None:
        """Sleep on the ROS clock (falls back to wall time)."""
        try:
            self._rospy.sleep(seconds)
        except Exception:  # noqa: BLE001
            time.sleep(seconds)
