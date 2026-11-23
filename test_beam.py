from beam import ss_center_point

beam = {
    "L": 2.0,
    "E": 200e9,
    "I": 1e-5,
    "P": 1000
}

positions = [0, 0.5, 1.0, 1.5, 2.0]

for x in positions:
    y = ss_center_point(x, beam)

    print(f"x = {x:.2f} m | Deflection = {y:.8f} m")