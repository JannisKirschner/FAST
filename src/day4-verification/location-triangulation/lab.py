# %% [markdown]
# # Location triangulation: where is that cluster, actually?
#
# Export controls are written about places. A licence says these accelerators may operate in this
# country and not that one, and the whole regime rests on someone being able to tell the difference
# after the crates have shipped. Serial numbers and paperwork travel with the hardware and can be
# rewritten; the one thing that cannot be rewritten is how long light takes to get there.
#
# So: a set of landmark servers at known locations sends a challenge to the chip and times the
# reply. Light in fibre covers about 200 km per millisecond, so a round trip of 20 ms means the
# responder is at most 2,000 km away — a hard bound from physics, not a claim anyone is trusting.
# Intersect enough of those bounds and you have a region. This is a live proposal for enforcing
# chip export controls, cheap enough to be plausible: a few dozen landmark servers and under a
# million dollars to run ([Brass and Aarne, 2025](https://www.iaps.ai/s/LocationVerificationforAIChips.pdf)),
# and it is the same constraint-based geolocation the networking literature has used for twenty
# years ([Gueye et al., 2006](https://doi.org/10.1109/TNET.2006.886332)).
#
# You'll build it, and it will localise a cluster to a region. Then you'll ask the two questions
# that decide whether that region is worth anything: how much can a dishonest operator move it, and
# is it small enough to name a country?
#
# **Duration:** 45 min. **Prerequisites:** Day 0. **GPU:** none — this is numpy and geometry.

# %%
# Installs the lab package on Colab; skipped when it's already importable (e.g. a local editable install).
try:
    import fast  # noqa: F401
except ImportError:
    # %pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast
    pass

# %%
import numpy as np

from fast.colab import setup
from fast.labs.day4_verification import location_triangulation as lab
from fast.testing import exercise

setup(require_gpu=False)

# %% [markdown]
# ## The setup
#
# Nine landmark servers around the region, and a cluster whose operator says it is in Singapore.
# It is in Johor Bahru, across the strait in Malaysia — about 25 km away, and in a different
# jurisdiction, which is the only fact the licence cares about.
#
# The round-trip times below are synthesised rather than measured. Each one is a great-circle
# distance inflated by its own path-stretch factor — fibre doesn't run in straight lines, and it
# detours by a different amount on every route — plus a jitter term. They sit in the right range for
# real inter-city latency. The geometry is the point, not the numbers.

# %%
truth = "Johor Bahru"
rtts = lab.measure_rtt(truth)
for name, rtt in sorted(rtts.items(), key=lambda item: item[1]):
    print(f"{name:>12}  {rtt:6.1f} ms round trip   ({lab.great_circle_km(lab.SITES[truth], lab.LANDMARKS[name]):>6,.0f} km away)")

# %% [markdown]
# ## Part 1: one landmark is a circle, several are a region

# %% [markdown]
# ### Exercise: turn a round trip into a distance bound
#
# The signal has to get there and come back, and it cannot beat the speed of light in fibre. Return
# the furthest away the responder could possibly be.


# %%
@exercise
def max_distance_km(rtt_ms, km_per_ms):
    """Largest distance in km a responder can be, given a round-trip time of `rtt_ms`.

    `km_per_ms` is how far the signal travels in one millisecond, one way. The round trip covers
    the distance twice. Anything the responder spends thinking only makes this bound looser, never
    tighter, so this is an upper bound and never a lower one.
    """
    return rtt_ms / 2 * km_per_ms


lab.check_max_distance_km(max_distance_km)

# %%
nearest = min(rtts, key=rtts.get)
print(f"{nearest} alone: the cluster is within {max_distance_km(rtts[nearest], lab.FIBRE_KM_PER_MS):,.0f} km")
print("which is a circle most of Southeast Asia sits inside")

# %% [markdown]
# ### Exercise: intersect the bounds
#
# Every landmark rules out everywhere too far from it to have answered that fast. What survives all
# nine is the region the verifier is left with. Take a grid of candidate locations and mark the ones
# that are consistent with every landmark at once.
#
# `lab.distances_to(points, landmark_point)` gives the distance from each grid point to one
# landmark, as an array.


# %%
@exercise
def feasible_mask(points, landmarks, rtts, km_per_ms):
    """Boolean array marking which `points` are consistent with every landmark's bound.

    `points` is an `(n, 2)` array of `(latitude, longitude)` candidates, `landmarks` maps a name to
    a `(latitude, longitude)` pair, and `rtts` maps the same names to round-trip times in ms. A
    point survives only if its distance to *every* landmark is within that landmark's bound. Return
    shape `(n,)`.
    """
    mask = np.ones(len(points), dtype=bool)
    for name, point in landmarks.items():
        mask &= lab.distances_to(points, point) <= max_distance_km(rtts[name], km_per_ms)
    return mask


lab.check_feasible_mask(feasible_mask)

# %%
points, shape = lab.grid()
region = feasible_mask(points, lab.LANDMARKS, rtts, lab.FIBRE_KM_PER_MS)
print(f"the verifier's region spans {lab.region_span_km(region, points):,.0f} km\n")
lab.show_region(region, points, shape)

# %% [markdown]
# The same region on a real map. Teal dots are the landmark servers, red markers the candidate
# sites, and the shaded cells are everywhere consistent with all nine measurements. Zoom into the
# strait at the bottom left.

# %%
lab.map_region(feasible_mask, rtts)

# %% [markdown]
# It worked, in the sense that nine numbers and some geometry cut the map down to a region. Whether
# it worked in the sense the licence needs is a different question, and the answer is on the map:
# ask which of the marked sites fall inside it.

# %%
for name in ("Singapore", "Johor Bahru", "Batam", "Kuala Lumpur", "Ho Chi Minh City", "Shenzhen"):
    inside = feasible_mask(np.array([lab.SITES[name]]), lab.LANDMARKS, rtts, lab.FIBRE_KM_PER_MS)[0]
    print(f"{name:>18}  {'consistent' if inside else 'ruled out'}")

# %% [markdown]
# Singapore, Johor Bahru, Batam and Kuala Lumpur are all consistent with the same measurements —
# four cities in three countries. The operator's claim to be in Singapore is true as far as this
# instrument can tell, and so is every other claim they might have made. Ho Chi Minh City and
# Shenzhen are genuinely excluded, so the measurement is not worthless; it answers a question about
# continents while the licence asks a question about countries.

# %% [markdown]
# ## Part 2: the operator can only ever add delay
#
# Before worrying about precision, settle whether this can be faked outright. The operator controls
# one thing: how long they wait before replying. They can stall, which makes them look further away.
# They cannot reply before the signal arrives, so they can never look closer.
#
# That one-sidedness is the mechanism's real guarantee, and it is worth being precise about what it
# buys. Every bound gets looser when the operator stalls, so the region only ever grows, and the
# true location never leaves it. A dishonest operator cannot move the verifier's region off the
# truth. They can only inflate it until it covers wherever they would like to be.

# %% [markdown]
# ### Exercise: what does a false claim cost?
#
# Work out the smallest delay the operator would have to add — to every reply, uniformly — before a
# claimed location becomes consistent with the measurements.


# %%
@exercise
def delay_to_claim(claimed, landmarks, rtts, km_per_ms):
    """Smallest uniform delay in ms that makes `claimed` consistent with every landmark.

    `claimed` is a `(latitude, longitude)` pair. For each landmark, the reported round trip has to
    be long enough to reach `claimed`; the operator can only add time, never remove it. Return the
    smallest delay that satisfies all of them at once, and 0.0 when the claim is already consistent.
    """
    return max(
        max(0.0, 2 * lab.great_circle_km(claimed, point) / km_per_ms - rtts[name])
        for name, point in landmarks.items()
    )


lab.check_delay_to_claim(delay_to_claim)

# %%
print(f"cluster really in {truth}\n")
for name in ("Singapore", "Batam", "Kuala Lumpur", "Ho Chi Minh City", "Shenzhen"):
    cost = delay_to_claim(lab.SITES[name], lab.LANDMARKS, rtts, lab.FIBRE_KM_PER_MS)
    verdict = "free — already consistent" if cost == 0 else f"{cost:.1f} ms of stalling"
    print(f"claiming {name:>18}: {verdict}")

# %% [markdown]
# The lie the operator actually wants to tell is free. They do not have to stall, spoof, or touch
# the network — Singapore is already inside the region, so the honest measurement supports the false
# claim on its own. The lies that cost something are the ones nobody would bother telling.
#
# And a cost in milliseconds is not much of a deterrent either. Stalling is invisible unless the
# verifier knows what the honest latency should have been, which is the thing they are trying to
# measure. A verifier can notice that a region has grown implausibly large and refuse to certify,
# which is a real defence — but it downgrades the mechanism from "proves where the chip is" to
# "notices when someone is obviously stalling".

# %% [markdown]
# ## Part 3: whose location did you measure?
#
# Everything so far assumed the thing that answered the challenge is the thing under licence. Take
# that away and none of the geometry survives. If the operator puts a small server in Singapore and
# has it answer the challenges while the accelerators run in Johor, every measurement in this
# notebook is a correct, honest, physically sound measurement of the location of a small server in
# Singapore.
#
# This is the same hole as Part 2 of the recomputation lab, in different clothing: the evidence is
# authored by the party being audited. Closing it needs the reply to be computed inside the chip
# itself, signed with a key that cannot be extracted, and fast enough that the response time is the
# network's rather than the operator's
# ([Petrie et al., 2025](https://arxiv.org/abs/2506.15093);
# [Aarne, Fist et al., 2024](https://www.cnas.org/publications/reports/secure-governable-chips)).
# Delay-based location verification is not a network technique with a hardware option bolted on. It
# is a hardware technique that happens to use the network as its clock.

# %% [markdown]
# ## Part 4: what sets the resolution
#
# Assume all of that solved — the chip answers for itself, honestly. How precisely can you place it?
#
# Two things put a floor under the answer, and they are worth separating. The first is geometry:
# a landmark 4,000 km away has a loose bound even when the measurement is perfect, so what you can
# resolve depends on your *nearest* landmark rather than on how many you have. Compare a cluster in
# Johor, whose closest landmark is Jakarta, with one in Shenzhen, which has Hong Kong 27 km away.

# %%
for site in ("Johor Bahru", "Shenzhen"):
    site_rtts = lab.measure_rtt(site)
    mask = feasible_mask(points, lab.LANDMARKS, site_rtts, lab.FIBRE_KM_PER_MS)
    closest = min(site_rtts, key=site_rtts.get)
    distance = lab.great_circle_km(lab.SITES[site], lab.LANDMARKS[closest])
    print(f"{site:>12}: region spans {lab.region_span_km(mask, points):>6,.0f} km   "
          f"(nearest landmark {closest}, {distance:,.0f} km away)")

# %% [markdown]
# A landmark next door is worth more than eight distant ones. But notice that Shenzhen still isn't
# pinned to anything like 27 km, and that is the second floor: measurement noise. A millisecond of
# jitter is 100 km of slop in the bound, whichever landmark it came from.
#
# Put that next to the distance the licence actually turns on.

# %%
separation = lab.great_circle_km(lab.SITES["Singapore"], lab.SITES["Johor Bahru"])
print(f"Singapore to Johor Bahru: {separation:.0f} km, a round trip of {2 * separation / lab.FIBRE_KM_PER_MS:.2f} ms")
print()
for jitter in (1.5, 0.5, 0.1):
    print(f"jitter of {jitter:>4} ms is worth {jitter / 2 * lab.FIBRE_KM_PER_MS:>3.0f} km of slop in every bound")

# %% [markdown]
# Drawn to scale, against the border it is supposed to resolve. Every site in the circle is
# indistinguishable from Singapore as far as this instrument is concerned.

# %%
lab.map_resolution(jitter_ms=1.5)

# %% [markdown]
# The border is a fifth of a millisecond wide. Nothing on the public internet is measured that
# precisely, and no amount of landmark density fixes it, because the noise is in the path rather
# than in the geometry. A mechanism that resolves to a hundred kilometres cannot answer a question
# posed about a seventeen-kilometre strait.
#
# There is one more way to tighten the region, and it is worth seeing why it is a worse idea than it
# looks. The bounds so far assume the signal might travel in a perfectly straight line at the speed
# of light in fibre. No real path does, so assuming a slower effective speed tightens every bound at
# once. The question is which assumption you are willing to make.

# %%
for label, assumed in (("physics only", 1.0), ("straightest seen", 1.3), ("typical", 1.5), ("optimistic", 1.7)):
    speed = lab.FIBRE_KM_PER_MS / assumed
    mask = feasible_mask(points, lab.LANDMARKS, rtts, speed)
    inside = feasible_mask(np.array([lab.SITES[truth]]), lab.LANDMARKS, rtts, speed)[0]
    if not mask.any():
        print(f"{label:>16} ({speed:5.1f} km/ms): no location satisfies every landmark — region empty")
        continue
    off_by = lab.great_circle_km(points[mask].mean(axis=0), lab.SITES[truth])
    print(f"{label:>16} ({speed:5.1f} km/ms): region {lab.region_span_km(mask, points):>6,.0f} km, "
          f"centred {off_by:>4,.0f} km from the truth, true location {'inside' if inside else 'EXCLUDED'}")

# %% [markdown]
# Read the middle two rows carefully, because that is where the damage is. The physics-only bound
# cannot exclude the truth, ever — that is what makes it a proof rather than an estimate.
# Calibrating to the straightest path anyone has observed keeps that property and buys real
# precision. Calibrating to a typical path buys more precision still, and the region it produces is
# tighter, plausible, non-empty, confidently centred a couple of hundred kilometres from where the
# cluster actually is, and wrong. Nothing in the output says so.
#
# The last row is the benign failure: assume too much and no location satisfies every landmark, so
# the verifier discovers their own calibration is broken. The dangerous setting is the one just
# before it, which looks like the best result on the page. Real schemes do calibrate, from
# measurements between landmarks of known position, and they inherit exactly this
# ([Gueye et al., 2006](https://doi.org/10.1109/TNET.2006.886332)).
#
# **Take it further, on your own time:**
#
# - Add a landmark in Kuala Lumpur and re-run Part 1. How much does the region shrink, and does it
#   now separate Singapore from Johor? What does that tell you about where a verifier has to be
#   allowed to put its servers, and who has to agree to it?
# - The operator adds delay to *some* landmarks and not others. Can they shape the region rather
#   than just inflate it — pull it towards a claimed location while keeping it small enough to look
#   like an honest measurement?
# - The jitter floor is the binding constraint at short range. What would a verifier have to control
#   — dedicated links, a hardware timestamp inside the chip, repeated measurements — to push it from
#   a millisecond to a microsecond, and which of those are things an operator could refuse?

# %% [markdown]
# ## What to take away
#
# The physics is sound and the guarantee is real: an operator can stall but cannot outrun light, so
# the region always contains the truth. That is a genuine one-sided proof, and it is rarer than it
# sounds — most verification mechanisms don't have one.
#
# It is also narrower than the question. The region contains the truth, and it contains a great deal
# else, and the operator gets to choose which of the places inside it to name. For a cluster in
# Johor Bahru, "we are in Singapore" costs nothing to say and cannot be contradicted. Continental
# questions — is this in East Asia or Europe — are answered well. Jurisdictional ones, which is what
# every export licence actually asks, are answered only where the border is wider than the noise.
#
# Three things generalise past this lab. The measurement locates whatever answered the challenge, so
# without a hardware root of trust it says nothing about the chips at all; the geometry is not the
# hard part, the binding is. The resolution is set by the nearest landmark and by the jitter floor,
# neither of which is a property of the protocol, which makes "where may we put landmark servers,
# and who consents to that" a treaty negotiation rather than an engineering decision. And the
# tempting fix — assume a realistic path speed rather than a physical one — trades the one-sided
# guarantee for precision, which is a bad trade for an instrument meant to support an accusation.
#
# A note on how this material usually gets framed: the worked examples in this literature are
# generally North American or European, where landmark density is high and the borders that matter
# are far apart. The Singapore–Johor–Batam triangle is three countries inside an hour's drive, and
# it is a real place people build datacentres. Whether a verification mechanism is fit for purpose
# is a question about a specific map, not a general one.

# %%
# @lab-only
# Stuck on feasible_mask? Start with np.ones(len(points), dtype=bool) and AND in one landmark at a
# time: lab.distances_to(points, point) <= max_distance_km(rtts[name], km_per_ms).
#
# Stuck on delay_to_claim? Per landmark, the round trip needed to reach the claim is
# 2 * great_circle_km(claimed, point) / km_per_ms. Subtract what the landmark already measured,
# clamp negatives to zero, and take the largest — every landmark has to be satisfied at once.
print("AND the landmarks together for the mask; take the max of the per-landmark shortfalls")
