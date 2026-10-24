#!/usr/bin/env python3
"""Find Baxter on a link without avahi-utils installed.

Sends mDNS queries with the unicast-response bit set, so answers come straight
back to our ephemeral port -- no need to bind 5353, which avahi-daemon already
owns. Prints every hostname/address it hears, Baxter-looking ones first.

    ./scripts/find-baxter.py            # 8 second scan
    ./scripts/find-baxter.py -t 20      # longer, for a slow-booting robot
"""

import argparse
import socket
import struct
import sys
import time

MCAST = "224.0.0.251"
PORT = 5353
QU = 0x8000  # ask for a unicast reply

PROBES = [
    ("_services._dns-sd._udp.local", 12),
    ("_workstation._tcp.local", 12),
    ("_ssh._tcp.local", 12),
    ("_udisks-ssh._tcp.local", 12),
]


def encode_name(name):
    out = b""
    for label in name.split("."):
        if label:
            out += bytes([len(label)]) + label.encode()
    return out + b"\x00"


def build_query(name, qtype):
    header = struct.pack("!HHHHHH", 0, 0, 1, 0, 0, 0)
    return header + encode_name(name) + struct.pack("!HH", qtype, QU | 1)


def read_name(buf, off):
    """Decode a DNS name, following compression pointers."""
    labels, jumped, safety = [], False, 0
    while safety < 128:
        safety += 1
        if off >= len(buf):
            break
        n = buf[off]
        if n == 0:
            off += 1
            break
        if n & 0xC0 == 0xC0:
            ptr = struct.unpack("!H", buf[off : off + 2])[0] & 0x3FFF
            if not jumped:
                end = off + 2
            off, jumped = ptr, True
            continue
        labels.append(buf[off + 1 : off + 1 + n].decode("utf-8", "replace"))
        off += 1 + n
    return ".".join(labels), (end if jumped else off)


def parse(buf):
    """Yield (name, rrtype, rdata, rdata_offset) for the answer sections."""
    try:
        _, _, qd, an, ns, ar = struct.unpack("!HHHHHH", buf[:12])
    except struct.error:
        return
    off = 12
    for _ in range(qd):
        _, off = read_name(buf, off)
        off += 4
    for _ in range(an + ns + ar):
        name, off = read_name(buf, off)
        if off + 10 > len(buf):
            return
        rrtype, _, _, rdlen = struct.unpack("!HHIH", buf[off : off + 10])
        off += 10
        rdata_off = off
        rdata = buf[off : off + rdlen]
        off += rdlen
        yield name, rrtype, rdata, rdata_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--timeout", type=float, default=8.0)
    ap.add_argument(
        "-i", "--iface-ip", default=None, help="source address to send from (e.g. 169.254.0.10)"
    )
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
    if args.iface_ip:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(args.iface_ip))
        sock.bind((args.iface_ip, 0))
    sock.settimeout(0.5)

    for name, qtype in PROBES:
        try:
            sock.sendto(build_query(name, qtype), (MCAST, PORT))
        except OSError as e:
            print("send failed: %s" % e, file=sys.stderr)

    hosts, services = {}, set()
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        try:
            buf, addr = sock.recvfrom(9000)
        except socket.timeout:
            continue
        except OSError:
            break
        for name, rrtype, rdata, rdata_off in parse(buf):
            if rrtype == 1 and len(rdata) == 4:  # A
                hosts.setdefault(socket.inet_ntoa(rdata), set()).add(name)
            elif rrtype == 12 and rdata:  # PTR
                target, _ = read_name(buf, rdata_off)
                if target:
                    services.add(target)
        hosts.setdefault(addr[0], set())

    if not hosts:
        print("Nothing answered mDNS in %.0fs." % args.timeout)
        print()
        print("If Baxter is booted and the link is up, it may not be running")
        print("avahi, or it may not be on this subnet. Next things to try:")
        print("  sudo tcpdump -i eno2 -n arp or udp port 5353   # watch it talk")
        print("  sudo apt install avahi-utils && avahi-browse -art")
        return 1

    def score(item):
        ip, names = item
        text = " ".join(names).lower()
        # Rethink names robots after the serial number, e.g. 011303P0017.local
        return (0 if ("baxter" in text or any(n[:1].isdigit() for n in names)) else 1, ip)

    print("Answered on the network:\n")
    for ip, names in sorted(hosts.items(), key=score):
        label = ", ".join(sorted(names)) if names else "(no name in reply)"
        print("  %-16s %s" % (ip, label))
    if services:
        print("\nAdvertised services:")
        for s in sorted(services):
            print("  %s" % s)
    print("\nPut the likely one in baxter.env as BAXTER_HOSTNAME (name or IP).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
