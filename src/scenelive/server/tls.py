"""A self-signed certificate for the phone to open https://<this computer>:8443 (browsers allow the camera only on https)."""
from __future__ import annotations
import datetime
import ipaddress
import os
import socket


def lan_ips() -> list[str]:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ips.append(s.getsockname()[0]); s.close()   # no packet is sent
    except OSError: pass
    try: ips += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError: pass
    out = []
    for ip in ips:
        if ip not in out and not ip.startswith("127."): out.append(ip)
    return out


def _covers(cert_path: str, names: list[str]) -> bool:
    from cryptography import x509
    try:
        cert = x509.load_pem_x509_certificate(open(cert_path, "rb").read())
        if cert.not_valid_after_utc < datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7): return False
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        have = {str(v) for v in san.get_values_for_type(x509.IPAddress)} | set(san.get_values_for_type(x509.DNSName))
        return all(n in have for n in names)
    except Exception: return False


def ensure_self_signed_cert(cert_dir: str, extra_names: list[str] | None = None) -> tuple[str, str]:
    """-> (cert.pem, key.pem). Reused while valid and covering this computer's current addresses; otherwise regenerated."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    os.makedirs(cert_dir, exist_ok=True); cp, kp = os.path.join(cert_dir, "cert.pem"), os.path.join(cert_dir, "key.pem")
    names = ["localhost", "127.0.0.1"] + lan_ips() + list(extra_names or [])
    if os.path.exists(cp) and os.path.exists(kp) and _covers(cp, names): return cp, kp
    key = ec.generate_private_key(ec.SECP256R1()); now = datetime.datetime.now(datetime.timezone.utc)
    san = []
    for n in names:
        try: san.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError: san.append(x509.DNSName(n))
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "scenelive")])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san), critical=False).add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    open(cp, "wb").write(cert.public_bytes(serialization.Encoding.PEM))
    open(kp, "wb").write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    try: os.chmod(kp, 0o600)
    except OSError: pass
    return cp, kp
