#!/usr/bin/env python3
import math


def decasteljau_split_1d(coeffs, lam):
    lam = max(0.0, min(1.0, float(lam)))
    points = [float(v) for v in coeffs]

    left = [points[0]]
    right = [points[-1]]

    while len(points) > 1:
        next_points = []
        for i in range(len(points) - 1):
            val = (1.0 - lam) * points[i] + lam * points[i + 1]
            next_points.append(val)

        left.append(next_points[0])
        right.append(next_points[-1])
        points = next_points

    right.reverse()
    pos = points[0]

    return left, right, pos


def sample_first_bernstein_segment(coeffs, lambda_end, num_points):
    segment_coeffs, rest_coeffs, cut_point = decasteljau_split_1d(coeffs, lambda_end)

    samples = []
    for i in range(num_points):
        mu = float(i) / float(num_points - 1)
        _, _, point = decasteljau_split_1d(segment_coeffs, mu)
        samples.append(point)

    return segment_coeffs, rest_coeffs, cut_point, samples


def main():
    px = [
        -1.200000000,
        -1.200000000,
        -1.182765000,
        -1.089802000,
        -0.524319000,
        0.681072000,
    ]

    py = [
        1.800000000,
        1.800000000,
        1.784570000,
        1.719352000,
        1.173538000,
        1.510945000,
    ]

    lambda_end = 0.30
    num_points = 20

    px_segment, px_rest, px_cut, px_samples = sample_first_bernstein_segment(
        px, lambda_end, num_points
    )

    py_segment, py_rest, py_cut, py_samples = sample_first_bernstein_segment(
        py, lambda_end, num_points
    )

    print("Original Bernstein control points:")
    for i, (x, y) in enumerate(zip(px, py)):
        print(f"CP {i:02d}: px = {x: .9f}, py = {y: .9f}")

    print("\nExtracted first segment:")
    print(f"lambda_end = {lambda_end:.6f}")
    print(f"percentage  = {100.0 * lambda_end:.2f}%")

    print("\nFinal point of extracted initial segment:")
    print(f"px_cut = {px_cut: .9f}")
    print(f"py_cut = {py_cut: .9f}")

    print("\nControl points of extracted initial segment:")
    for i, (x, y) in enumerate(zip(px_segment, py_segment)):
        print(f"SEG CP {i:02d}: px = {x: .9f}, py = {y: .9f}")

    print("\n20 sampled points inside extracted initial segment:")
    for i, (x, y) in enumerate(zip(px_samples, py_samples)):
        mu = float(i) / float(num_points - 1)
        print(f"P {i:02d}: mu = {mu:.6f}, px = {x: .9f}, py = {y: .9f}")

    print("\nRest of trajectory control points after the cut:")
    for i, (x, y) in enumerate(zip(px_rest, py_rest)):
        print(f"REST CP {i:02d}: px = {x: .9f}, py = {y: .9f}")


if __name__ == "__main__":
    main()
