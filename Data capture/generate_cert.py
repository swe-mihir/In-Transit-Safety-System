"""
generate_cert.py
================
Generates a self-signed SSL certificate (cert.pem + key.pem) in the
current directory, valid for your local IP and localhost.

Run once before starting server.py:
    pip install cryptography
    python generate_cert.py
"""

import datetime
import ipaddress
import socket

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

# ── Detect local IP ───────────────────────────────────────────────────────────
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
except Exception:
    local_ip = "127.0.0.1"
finally:
    s.close()

print(f"Detected local IP: {local_ip}")
print("Generating 2048-bit RSA key and self-signed certificate ...")

# ── Generate private key ──────────────────────────────────────────────────────
key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend(),
)

# ── Build certificate ─────────────────────────────────────────────────────────
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, local_ip)])

cert = (
    x509.CertificateBuilder()
    .subject_name(name)
    .issuer_name(name)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=825))
    .add_extension(
        x509.SubjectAlternativeName([
            x509.IPAddress(ipaddress.IPv4Address(local_ip)),
            x509.DNSName("localhost"),
        ]),
        critical=False,
    )
    .sign(key, hashes.SHA256(), default_backend())
)

# ── Write files ───────────────────────────────────────────────────────────────
with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

with open("key.pem", "wb") as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))

print()
print("  cert.pem  — certificate (valid 825 days)")
print("  key.pem   — private key")
print()
print(f"  Certificate covers: {local_ip}  and  localhost")
print()
print("Next step:  python server.py")
