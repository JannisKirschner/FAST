"""Fixtures and checks for `src/day4-verification/location-triangulation`.

Delay-based geolocation: a landmark server challenges a chip, times the reply, and turns the
round-trip into an upper bound on how far away the chip can be. Intersect enough of those bounds
and you have a region. The lab builds that, then asks what it actually establishes.

Round-trip times here are synthesised from great-circle distance with a path-stretch factor and
jitter, not measured. The numbers land in the right range for inter-city internet latency, but the
lab is about the geometry and the one-sidedness of the constraint, not about any real network.

Pure numpy and CPU-only: no model, no GPU, runs in CI in under a second.
"""

from __future__ import annotations

import numpy as np

from fast.testing import checker, require

__all__ = [
    "FIBRE_KM_PER_MS",
    "LANDMARKS",
    "SITES",
    "STRETCH",
    "STRETCH_RANGE",
    "check_delay_to_claim",
    "check_feasible_mask",
    "check_max_distance_km",
    "distances_to",
    "great_circle_km",
    "grid",
    "measure_rtt",
    "plot_region",
    "plot_resolution",
    "region_span_km",
    "show_region",
]

# Light in glass, one way. c/1.47 is about 204,000 km/s, so roughly 200 km per millisecond — and a
# round trip covers the distance twice.
FIBRE_KM_PER_MS = 200.0

# Real paths are not great circles. Fibre follows coastlines, cables land where they land, and
# routing sends a packet through whichever exchange the operators peer at, so measured latency runs
# well above the straight-line minimum — and by a different amount on every path, which is what
# makes calibrating it dangerous. Each landmark's path gets its own stretch from this range.
STRETCH_RANGE = (1.3, 2.1)

# The straightest path anyone has seen. A verifier calibrating from known-position measurements
# reads off this lower envelope, because assuming anything tighter can exclude an honest operator.
STRETCH = STRETCH_RANGE[0]

# Landmark servers: hosts at known locations that issue the challenges. Kept to cities a verifier
# could plausibly place one in, and none of them next door to the sites under test.
LANDMARKS = {
    "Bangkok": (13.76, 100.50),
    "Jakarta": (-6.21, 106.85),
    "Manila": (14.60, 120.98),
    "Hong Kong": (22.32, 114.17),
    "Taipei": (25.03, 121.57),
    "Seoul": (37.57, 126.98),
    "Tokyo": (35.68, 139.69),
    "Perth": (-31.95, 115.86),
    "Mumbai": (19.08, 72.88),
}

# Candidate locations for the cluster. The first three sit inside an hour of each other and in
# three different countries, which is the whole question an export-control regime has to answer.
SITES = {
    "Singapore": (1.35, 103.82),
    "Johor Bahru": (1.49, 103.76),
    "Batam": (1.08, 104.03),
    "Kuala Lumpur": (3.14, 101.69),
    "Ho Chi Minh City": (10.82, 106.63),
    "Shenzhen": (22.54, 114.06),
}


def great_circle_km(a, b) -> float:
    """Distance in km between two `(latitude, longitude)` points, in degrees."""
    lat1, lon1, lat2, lon2 = np.radians([a[0], a[1], b[0], b[1]])
    haversine = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(haversine)))


def measure_rtt(site, landmarks=None, *, jitter_ms: float = 1.5, added_delay_ms: float = 0.0,
                seed: int = 0) -> dict[str, float]:
    """Round-trip times in ms from each landmark to a cluster at `site`.

    `site` is a name in `SITES` or a `(latitude, longitude)` pair. Each landmark's straight-line
    round trip is inflated by its own stretch factor drawn from `STRETCH_RANGE` — one path runs
    almost straight, another detours through a distant exchange — then nudged by `jitter_ms` of
    noise. `added_delay_ms` is whatever the operator chose to sit on before replying.
    """
    landmarks = LANDMARKS if landmarks is None else landmarks
    location = SITES[site] if isinstance(site, str) else site
    rng = np.random.default_rng(seed)
    return {
        name: 2 * great_circle_km(location, point) / FIBRE_KM_PER_MS * rng.uniform(*STRETCH_RANGE)
        + abs(rng.normal(scale=jitter_ms))
        + added_delay_ms
        for name, point in landmarks.items()
    }


def grid(step_deg: float = 0.5):
    """A lat/lon grid over South and East Asia: `(points, shape)` for `(n, 2)` points."""
    lats = np.arange(-35.0, 45.0 + step_deg, step_deg)
    lons = np.arange(65.0, 155.0 + step_deg, step_deg)
    mesh_lat, mesh_lon = np.meshgrid(lats, lons, indexing="ij")
    return np.stack([mesh_lat.ravel(), mesh_lon.ravel()], axis=1), (len(lats), len(lons))


def distances_to(points, landmark_point) -> np.ndarray:
    """Distance in km from every `(n, 2)` lat/lon point to one landmark."""
    lat1, lon1 = np.radians(points[:, 0]), np.radians(points[:, 1])
    lat2, lon2 = np.radians(landmark_point[0]), np.radians(landmark_point[1])
    haversine = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(haversine))


def region_span_km(mask, points) -> float:
    """Greatest distance between any two points the verifier still considers possible."""
    inside = points[np.asarray(mask, dtype=bool)]
    if len(inside) < 2:
        return 0.0
    corners = [(inside[:, 0].min(), inside[:, 1].min()), (inside[:, 0].max(), inside[:, 1].max()),
               (inside[:, 0].min(), inside[:, 1].max()), (inside[:, 0].max(), inside[:, 1].min())]
    return max(great_circle_km(corners[0], corners[1]), great_circle_km(corners[2], corners[3]))


def show_region(mask, points, shape, sites=("Singapore", "Johor Bahru", "Shenzhen"), cols: int = 72) -> None:
    """Print the feasible region as a map, with the named sites marked."""
    mask = np.asarray(mask, dtype=bool).reshape(shape)
    rows = max(1, round(cols * shape[0] / shape[1] / 2.2))
    row_edges = np.linspace(0, shape[0], rows + 1).astype(int)
    col_edges = np.linspace(0, shape[1], cols + 1).astype(int)
    canvas = [["#" if mask[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]].any() else "·"
               for c in range(cols)] for r in range(rows)]

    lats, lons = points[:, 0].reshape(shape)[:, 0], points[:, 1].reshape(shape)[0]
    placed, collisions = {}, []
    for marker, name in zip("ABCDEF", sites):
        lat, lon = SITES[name]
        r = min(rows - 1, np.searchsorted(row_edges, np.abs(lats - lat).argmin(), "right") - 1)
        c = min(cols - 1, np.searchsorted(col_edges, np.abs(lons - lon).argmin(), "right") - 1)
        if (r, c) in placed:
            collisions.append(f"{placed[(r, c)]} and {name}")
            continue
        placed[(r, c)] = name
        canvas[r][c] = marker

    print(f"latitude {lats[-1]:.0f}N to {lats[0]:.0f}N, longitude {lons[0]:.0f}E to {lons[-1]:.0f}E")
    for row in reversed(canvas):
        print("".join(row))
    print("  ".join(f"{marker}={name}" for marker, name in zip("ABCDEF", sites) if name in placed.values()))
    for pair in collisions:
        print(f"({pair} land in the same cell at this scale — which is rather the point)")


def _borders(name: str):
    """Country outlines bundled with the package, as a list of (lons, lats) rings.

    Natural Earth, public domain, shipped in the wheel rather than downloaded. The lab drew on
    hosted map tiles once and was blocked for it — a room of twenty people pulling from
    volunteer-run tile servers is what the OpenStreetMap tile usage policy exists to stop, and the
    commercial tile providers want an API key. Vectors in the package need neither.
    """
    import json
    from importlib.resources import files

    source = files("fast.labs.day4_verification").joinpath("borders", name).read_text()
    rings = []
    for feature in json.loads(source)["features"]:
        for polygon in feature["geometry"]["coordinates"]:
            ring = np.array(polygon[0])
            rings.append((ring[:, 0], ring[:, 1]))
    return rings


def _basemap(ax, rings, extent):
    ax.set_facecolor("#d9e6f2")  # sea, so land reads as land without a legend
    for lons, lats in rings:
        ax.fill(lons, lats, facecolor="#f7f5f1", edgecolor="#7d8a97", linewidth=0.6, zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    # Equirectangular: stretch longitude by cos(latitude) so the shapes aren't squashed.
    ax.set_aspect(1 / np.cos(np.radians(np.mean(extent[2:]))))
    ax.set_xlabel("longitude"), ax.set_ylabel("latitude")
    ax.grid(color="#ffffff", alpha=0.6, linewidth=0.5, zorder=0)


def _geodesic_circle(centre, radius_km, points: int = 361):
    """The set of points exactly `radius_km` from `centre`, on the sphere rather than on the page."""
    lat0, lon0 = np.radians(centre[0]), np.radians(centre[1])
    angular = radius_km / 6371.0
    bearings = np.linspace(0, 2 * np.pi, points)
    lat = np.arcsin(np.sin(lat0) * np.cos(angular) + np.cos(lat0) * np.sin(angular) * np.cos(bearings))
    lon = lon0 + np.arctan2(
        np.sin(bearings) * np.sin(angular) * np.cos(lat0), np.cos(angular) - np.sin(lat0) * np.sin(lat)
    )
    return np.degrees(lon), np.degrees(lat)


def plot_region(feasible, rtts, *, step_deg: float = 0.5, sites=("Singapore", "Johor Bahru", "Batam")):
    """Draw a feasible region over country borders, with landmarks and sites marked.

    `feasible` is your own `feasible_mask`. Borders are the point of the background here rather
    than decoration: the question the licence asks is which country the cluster is in.
    """
    import matplotlib.pyplot as plt

    points, shape = grid(step_deg)
    mask = np.asarray(feasible(points, LANDMARKS, rtts, FIBRE_KM_PER_MS), dtype=bool)
    lats, lons = points[:, 0].reshape(shape)[:, 0], points[:, 1].reshape(shape)[0]

    figure, ax = plt.subplots(figsize=(11, 8))
    _basemap(ax, _borders("borders_region.geojson"), (lons[0], lons[-1], lats[0], lats[-1]))
    ax.contourf(lons, lats, mask.reshape(shape), levels=[0.5, 1.5], colors=["#c026d3"], alpha=0.3, zorder=1)

    for name, (lat, lon) in LANDMARKS.items():
        ax.plot(lon, lat, "o", color="#0f766e", markersize=6, zorder=3)
        ax.annotate(name, (lon, lat), (4, 4), textcoords="offset points", fontsize=8, color="#0f766e")
    clusters = []
    for name in sites:
        lat, lon = SITES[name]
        ax.plot(lon, lat, "*", color="#b91c1c", markersize=13, zorder=4)
        near = next((c for c in clusters if abs(c["lat"] - lat) < 2 and abs(c["lon"] - lon) < 2), None)
        if near:
            near["names"].append(name)
        else:
            clusters.append({"lat": lat, "lon": lon, "names": [name]})
    for cluster in clusters:
        ax.annotate(" · ".join(cluster["names"]), (cluster["lon"], cluster["lat"]), (8, -12),
                    textcoords="offset points", fontsize=8, color="#b91c1c", fontweight="bold")

    ax.set_title(f"Consistent with all {len(LANDMARKS)} landmarks (teal) — candidate sites in red")
    figure.tight_layout()
    print(f"{int(mask.sum())} of {mask.size} grid cells survive every landmark's bound")
    return figure


def plot_resolution(jitter_ms: float = 1.5, sites=("Singapore", "Johor Bahru", "Batam"), centre="Singapore"):
    """Draw what a millisecond of jitter is worth against the border it has to resolve."""
    import matplotlib.pyplot as plt

    slop_km = jitter_ms / 2 * FIBRE_KM_PER_MS
    lon, lat = _geodesic_circle(SITES[centre], slop_km)
    pad = 0.35
    extent = (lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad)

    figure, ax = plt.subplots(figsize=(9, 8))
    _basemap(ax, _borders("borders_strait.geojson"), extent)
    ax.fill(lon, lat, facecolor="#c026d3", alpha=0.18, edgecolor="#c026d3", linewidth=1.5, zorder=2)

    for name in sites:
        site_lat, site_lon = SITES[name]
        away = great_circle_km(SITES[centre], SITES[name])
        ax.plot(site_lon, site_lat, "*", color="#b91c1c", markersize=15, zorder=4)
        ax.annotate(f"{name} ({away:.0f} km)", (site_lon, site_lat), (7, 5), textcoords="offset points",
                    fontsize=9, color="#b91c1c", fontweight="bold")

    ax.set_title(f"{slop_km:.0f} km — what {jitter_ms} ms of jitter is worth, over three countries")
    figure.tight_layout()
    print(f"{jitter_ms} ms of jitter is worth {slop_km:.0f} km; every site inside the shape is "
          f"indistinguishable from {centre}")
    return figure


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


@checker("max_distance_km")
def check_max_distance_km(fn) -> None:
    require(np.isclose(fn(10.0, 200.0), 1000.0), f"10 ms at 200 km/ms is 1000 km away, got {fn(10.0, 200.0)}")
    require(np.isclose(fn(0.0, 200.0), 0.0), f"a zero round trip is zero distance, got {fn(0.0, 200.0)}")
    require(np.isclose(fn(1.0, 300.0), 150.0), f"1 ms at 300 km/ms is 150 km, got {fn(1.0, 300.0)}")
    doubled = fn(20.0, 200.0)
    require(
        np.isclose(doubled, 2000.0),
        f"twice the round trip is twice the distance: expected 2000 km, got {doubled} — "
        "the round trip covers the distance twice, so halve it before converting",
    )


@checker("feasible_mask")
def check_feasible_mask(fn) -> None:
    # 1 degree of longitude at the equator is ~111 km; a 12 ms round trip allows 1200 km. The point
    # at 15 degrees (~1670 km) is the one that matters: it is outside the correct bound and inside a
    # bound computed without halving the round trip, so it catches that mistake on its own.
    points = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 10.0], [0.0, 15.0], [0.0, 50.0]])
    landmarks = {"origin": (0.0, 0.0)}
    mask = np.asarray(fn(points, landmarks, {"origin": 12.0}, FIBRE_KM_PER_MS), dtype=bool)
    require(mask.shape == (5,), f"expected one boolean per point, shape (5,), got {mask.shape}")
    require(
        list(mask) == [True, True, True, False, False],
        f"1200 km reaches the first three points and neither of the last two, got {list(mask)} — "
        "the round trip covers the distance twice, so halve it before converting to km",
    )

    two = fn(points, {"origin": (0.0, 0.0), "east": (0.0, 50.0)}, {"origin": 12.0, "east": 12.0}, FIBRE_KM_PER_MS)
    require(
        list(np.asarray(two, dtype=bool)) == [False] * 5,
        "a point has to satisfy *every* landmark's bound, not any of them — "
        f"nothing here is within 1200 km of both, got {list(np.asarray(two, dtype=bool))}",
    )

    generous = fn(points, landmarks, {"origin": 1000.0}, FIBRE_KM_PER_MS)
    require(bool(np.all(np.asarray(generous, dtype=bool))), "a huge round trip should rule nothing out")


@checker("delay_to_claim")
def check_delay_to_claim(fn) -> None:
    landmarks = {"origin": (0.0, 0.0)}
    honest = {"origin": 12.0}  # allows 1200 km

    near = fn((0.0, 1.0), landmarks, honest, FIBRE_KM_PER_MS)
    require(
        np.isclose(near, 0.0),
        f"a claim already inside the honest region costs nothing to make, expected 0.0, got {near}",
    )

    far = fn((0.0, 50.0), landmarks, honest, FIBRE_KM_PER_MS)
    distance = great_circle_km((0.0, 0.0), (0.0, 50.0))
    expected = 2 * distance / FIBRE_KM_PER_MS - 12.0
    require(
        np.isclose(far, expected, rtol=1e-6),
        f"reaching {distance:,.0f} km needs a {2 * distance / FIBRE_KM_PER_MS:.1f} ms round trip, "
        f"so {expected:.1f} ms on top of the honest {honest['origin']:.1f} — got {far}",
    )

    two = fn((0.0, 50.0), {"origin": (0.0, 0.0), "east": (0.0, 60.0)},
             {"origin": 12.0, "east": 12.0}, FIBRE_KM_PER_MS)
    require(
        two >= expected - 1e-9,
        "every landmark's bound has to reach the claim, so the cost is the largest of the "
        f"per-landmark delays, not the smallest or the mean — expected at least {expected:.1f}, got {two}",
    )

    require(
        fn((0.0, 60.0), landmarks, honest, FIBRE_KM_PER_MS) > far,
        "a further claim costs more delay — check the direction",
    )
