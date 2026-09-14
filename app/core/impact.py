"""
impact.py
---------
Turns route-optimisation results into real-world impact: fuel, CO2, driver
hours, and what those become at city scale.

Resolves issue #12. The reasoning there is that "QPSO beats GA by X%" means
nothing to a non-technical judge, while "this saves ~Y litres of fuel and Z kg
CO2 per 100 deliveries" lands immediately.

Single source of truth
======================
Every conversion factor lives in `ASSUMPTIONS` below. Before this module the
dashboard carried two different CO2 factors — a KPI card used 0.22 kg/km while
the impact panel used 8 km/L x 2.68 kg/L = 0.335 kg/km — so the same run showed
two numbers 52% apart on one screen. Anything that converts distance or time
into impact must use these values, and `test_impact.py` asserts the dashboard's
copy still matches.

Honesty rules
=============
These carry over from the rest of the project:

  * A measured quantity and an assumed one are labelled differently. Distance
    saved is measured; litres of fuel is distance divided by an assumed fuel
    economy, and the assumption travels with the number.
  * Nothing is clamped. If the optimiser drives further, the fuel figure is
    negative and says so — on real road networks that happens routinely,
    because going around a jam beats queueing in it.
  * A city-scale figure is a projection, never a measurement, and is marked as
    one wherever it is produced.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ImpactAssumptions:
    """
    Conversion factors, with sources. Every one of these is arguable; they are
    stated explicitly so a judge can challenge the number rather than the claim.
    """

    # Diesel light commercial vehicle in stop-start urban traffic. Manufacturer
    # figures for this class run 10-14 km/L on a highway cycle; urban delivery
    # duty with frequent idling is materially worse, so 8 is a deliberately
    # conservative working value.
    km_per_litre: float = 8.0

    # DEFRA / IPCC combustion factor for diesel: burning one litre releases
    # about 2.68 kg of CO2. This one is physics and is not really arguable.
    kg_co2_per_litre: float = 2.68

    # Working days per year, for annualising a daily saving. 250 is the usual
    # business-day count (52 weeks x 5, less public holidays).
    working_days_per_year: int = 250

    # Used only to describe a "per 100 deliveries" figure, which is the unit
    # issue #12 asks for.
    deliveries_basis: int = 100

    def kg_co2_per_km(self) -> float:
        """Emissions per kilometre implied by the two factors above."""
        return self.kg_co2_per_litre / self.km_per_litre

    def describe(self) -> str:
        return (
            f"Assumes {self.km_per_litre:g} km per litre (diesel delivery van, urban "
            f"duty cycle) and {self.kg_co2_per_litre:g} kg CO2 per litre burned "
            f"(DEFRA factor) — {self.kg_co2_per_km():.3f} kg CO2 per km. "
            f"Annual figures assume {self.working_days_per_year} working days."
        )


ASSUMPTIONS = ImpactAssumptions()


@dataclass
class ImpactResult:
    """
    Impact of one optimisation run.

    Signed throughout: negative means the optimised plan did worse on that
    measure. `measured` separates what came from the solver from what came from
    a conversion factor, so a slide can cite them differently.
    """

    # measured directly by the solver
    km_saved: float
    minutes_saved: float
    deliveries: int

    # derived using ASSUMPTIONS
    litres_saved: float
    kg_co2_saved: float
    driver_hours_saved: float

    # normalised to the unit issue #12 asks for
    litres_per_100_deliveries: float
    kg_co2_per_100_deliveries: float

    assumptions_note: str
    measured_fields: tuple = field(
        default=("km_saved", "minutes_saved", "deliveries"), repr=False
    )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("measured_fields", None)
        return d


def compute_impact(
    km_saved: float,
    minutes_saved: float,
    deliveries: int,
    assumptions: Optional[ImpactAssumptions] = None,
) -> ImpactResult:
    """
    Convert a measured saving into fuel, CO2 and driver time.

    `km_saved` and `minutes_saved` are signed: pass a negative value when the
    optimised plan drove further or took longer, and the derived figures stay
    negative rather than being clamped to zero.
    """
    a = assumptions or ASSUMPTIONS

    litres = km_saved / a.km_per_litre
    co2 = litres * a.kg_co2_per_litre
    hours = minutes_saved / 60.0

    if deliveries > 0:
        per100 = a.deliveries_basis / deliveries
        litres_per_100 = litres * per100
        co2_per_100 = co2 * per100
    else:
        litres_per_100 = 0.0
        co2_per_100 = 0.0

    return ImpactResult(
        km_saved=round(km_saved, 2),
        minutes_saved=round(minutes_saved, 2),
        deliveries=deliveries,
        litres_saved=round(litres, 2),
        kg_co2_saved=round(co2, 2),
        driver_hours_saved=round(hours, 2),
        litres_per_100_deliveries=round(litres_per_100, 2),
        kg_co2_per_100_deliveries=round(co2_per_100, 2),
        assumptions_note=a.describe(),
    )


@dataclass
class CityScaleProjection:
    """A city-wide extrapolation. A projection, never a measurement."""

    fleet_size: int
    vehicles_in_run: int
    working_days: int
    annual_litres: float
    annual_kg_co2: float
    annual_tonnes_co2: float
    annual_driver_hours: float
    annual_km: float
    caveat: str

    def to_dict(self) -> Dict:
        return asdict(self)


def project_to_city(
    impact: ImpactResult,
    vehicles_in_run: int,
    fleet_size: int = 100,
    assumptions: Optional[ImpactAssumptions] = None,
) -> CityScaleProjection:
    """
    Scale one fleet's daily saving to a city fleet over a working year.

    `vehicles_in_run` must be the real vehicle count from the run. Guessing it
    (defaulting to 1, say) multiplies the projection several times over, so a
    non-positive value raises rather than quietly inflating the result.
    """
    a = assumptions or ASSUMPTIONS
    if vehicles_in_run <= 0:
        raise ValueError(
            "vehicles_in_run must be positive — without the real fleet size the "
            "scaling factor is unknown and the projection would be fabricated."
        )

    factor = (fleet_size / vehicles_in_run) * a.working_days_per_year
    annual_co2 = impact.kg_co2_saved * factor

    return CityScaleProjection(
        fleet_size=fleet_size,
        vehicles_in_run=vehicles_in_run,
        working_days=a.working_days_per_year,
        annual_litres=round(impact.litres_saved * factor, 1),
        annual_kg_co2=round(annual_co2, 1),
        annual_tonnes_co2=round(annual_co2 / 1000.0, 2),
        annual_driver_hours=round(impact.driver_hours_saved * factor, 1),
        annual_km=round(impact.km_saved * factor, 1),
        caveat=(
            f"Projection, not a measurement: {vehicles_in_run} vehicle(s) measured, "
            f"scaled to {fleet_size} over {a.working_days_per_year} working days. "
            f"Assumes every vehicle sees similar traffic and a similar workload, "
            f"which a real city will not match exactly."
        ),
    )
