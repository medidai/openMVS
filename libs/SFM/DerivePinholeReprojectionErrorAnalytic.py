import sympy as sp

"""
This script derives the analytic Jacobians for the PinholeReprojectionErrorAnalytic
cost function in OpenMVS.

USAGE:
1. Ensure SymPy is installed: `pip install sympy`
2. Run the script: `python DerivePinholeReprojectionErrorAnalytic.py`
3. The output provides the C-code for the Jacobian of the camera-space point (P_cam)
   with respect to the quaternion (q).

MAPPING TO C++:
The output `J_Pcam_q[i][j]` maps directly to the intermediate `J_Pcam_q` matrix
in `PinholeReprojectionErrorAnalytic::Evaluate`.
The full Jacobian is computed via the chain rule:
J_full = J_pixel_Pcam * J_Pcam_q

where:
- `J_pixel_Pcam` is the Jacobian of the full projection/distortion model w.r.t camera-space point.
- `J_Pcam_q` is the Jacobian of the world-to-camera transform w.r.t pose.

DERIVATION LOGIC:
It uses a homogeneous degree-2 rotation formula derivative:
f(q) = (w^2 + |v|^2)p + 2w(v x p) + 2(v x (v x p))

This formula is scale-invariant (f(kq) = k^2 f(q)), so after perspective division (x/z),
the result is invariant to the scale of the quaternion. This ensures that the
analytic Jacobian matches the manifold-aware numerical and auto-diff derivatives
without requiring explicit normalization in the cost function.
"""

def DerivePinholeReprojectionErrorAnalytic():
    # --- 1. Parameters ---
    # Quaternion and relative point
    qw, qx, qy, qz = sp.symbols('qw qx qy qz')
    v = sp.Matrix([qx, qy, qz])
    dx, dy, dz = sp.symbols('dx dy dz') # World point - Camera center
    p = sp.Matrix([dx, dy, dz])

    # Intrinsics
    fx, ar, cx, cy = sp.symbols('fx ar cx cy')
    k1, k2, k3, p1, p2, k4, k5, k6 = sp.symbols('k1 k2 k3 p1 p2 k4 k5 k6')

    def cross(a, b):
        return sp.Matrix([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ])

    # --- 2. Camera Space Transformation (Homogeneous Rotation) ---
    # R(q)*p = (w^2 + |v|^2)p + 2w(v x p) + 2*(v x (v x p))
    # Note: 2*(v x (v x p)) = 2*(v(v.p) - p(v.v))
    uv = 2 * cross(v, p)
    P_cam = (qw**2 + qx**2 + qy**2 + qz**2) * p + qw * uv + cross(v, uv)

    # --- 3. Perspective Projection ---
    xc, yc, zc = P_cam[0], P_cam[1], P_cam[2]
    un = xc / zc
    vn = yc / zc

    # --- 4. Distortion ---
    r2 = un**2 + vn**2
    r4 = r2**2
    r6 = r4*r2

    num = (1 + k1*r2 + k2*r4 + k3*r6)
    den = (1 + k4*r2 + k5*r4 + k6*r6)
    radial = num / den

    ud = un * radial + 2*p1*un*vn + p2*(r2 + 2*un**2)
    vd = vn * radial + p1*(r2 + 2*vn**2) + 2*p2*un*vn

    # --- 5. Pixel Mapping ---
    u = fx * ud + cx
    v_pixel = fx * ar * vd + cy # ar = fy/fx

    residuals = sp.Matrix([u, v_pixel])

    # --- 6. Jacobian Derivation ---
    q_params = [qw, qx, qy, qz]
    pt_params = [dx, dy, dz]
    # We focus on the P_cam part for the pose/point chain rule
    # but the script could derive the full thing if needed.

    J_Pcam_q = P_cam.jacobian(q_params)

    # --- Printing ---
    print("// Consolidated Analytic Jacobian Derivations")
    print("// 1. Camera Space Point P_cam w.r.t Quaternion q")
    for i in range(3):
        for j in range(4):
            expr = sp.simplify(J_Pcam_q[i, j])
            print(f"J_Pcam_q[{i}][{j}] = {sp.ccode(expr)};")

    # Note: J_Pcam_C = -R(q)
    # Note: J_Pcam_X = R(q)

if __name__ == "__main__":
    DerivePinholeReprojectionErrorAnalytic()
