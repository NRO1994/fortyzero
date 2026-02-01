# Wind Volume Model Specification

**Version:** 1.0
**Date:** 2026-02-01
**Branch:** `feature/wind_volume_model`
**Author:** Claude Code

---

## 1. Executive Summary

This specification describes the implementation of a `WindVolumeModel` for calculating energy production of wind parks. The model supports both **onshore and offshore** wind farms in a single implementation, using configurable parameters to account for different environmental conditions and wake effects.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Onshore/Offshore | **Single unified model** | Same physics, different parameters (wake decay, turbulence) |
| Wake Model | **Jensen (N.O. Jensen 1983)** | Simple, fast, well-validated, vectorizable |
| Temporal Resolution | **Hourly** | Consistent with PVVolumeModel, enables price curve integration |
| Wind Data Source | **Wind speed time series** | User-provided or from external APIs (future) |

---

## 2. Research Summary

### 2.1 Onshore vs. Offshore Differences

Based on research from [EuroWindWakes Project](https://www.offshorewind.biz/2025/05/26/new-research-project-aims-to-create-more-accurate-wake-modelling-for-offshore-wind/) and [academic literature](https://www.rees-journal.org/articles/rees/full_html/2021/01/rees210047/rees210047.html):

| Parameter | Onshore | Offshore |
|-----------|---------|----------|
| Wake decay coefficient (k) | 0.075 | 0.04-0.05 |
| Turbulence intensity | 10-20% | 5-8% |
| Turbine spacing (perpendicular) | 3-4 D | 4-9 D |
| Turbine spacing (row) | 4-6 D | 6-11 D |
| Wake recovery | Faster (higher turbulence) | Slower (lower turbulence) |
| Typical wake losses | 10-20% | 15-40% |

**Conclusion:** The core physics are identical; only parameters differ. A single model with site-type configuration is appropriate.

### 2.2 Wake Effect Models

Comparison from [ScienceDirect review](https://www.sciencedirect.com/science/article/abs/pii/S030626191830802X):

| Model | Complexity | Accuracy | Computation Time |
|-------|------------|----------|------------------|
| Jensen | Low | Good (general) | Very fast (<1ms) |
| Frandsen | Medium | Underestimates | Fast |
| Bastankhah & Porté-Agel | Medium | Better near-wake | Moderate |
| Larsen | Medium | Good | Moderate |
| CFD (LES/RANS) | Very High | Best | Hours |

**Decision:** Jensen model is optimal for our <10ms constraint while providing reasonable accuracy (within 10-15% of measured data per [WES Copernicus](https://wes.copernicus.org/articles/7/2407/2022/)).

### 2.3 Maintenance & Availability

Based on [DNV GL White Paper](https://www.ourenergypolicy.org/wp-content/uploads/2017/08/Definitions-of-availability-terms-for-the-wind-industry-white-paper-09-08-2017.pdf):

| Availability Type | Definition |
|-------------------|------------|
| Technical Availability | Time turbine can generate when wind is in operating range |
| Time-Based Availability | Total operational hours / Total hours |
| Production-Based Availability | Actual production / Theoretical production |

**Maintenance Categories:**
- **Scheduled (Preventive):** Planned inspections, oil changes, blade cleaning
- **Unscheduled (Reactive):** Component failures, emergency repairs
- **Condition-Based:** Triggered by sensor data (vibration, temperature)

---

## 3. Architecture

### 3.1 Class Diagram

```
WindVolumeModel
├── calculate(params) -> np.ndarray  # Hourly production (interface method)
├── calculate_monthly(params) -> np.ndarray  # Convenience aggregation
├── get_annual_production(params) -> np.ndarray
│
├── _calculate_single_turbine_power(wind_speed, power_curve, air_density)
├── _calculate_wake_deficit(distance, rotor_diameter, thrust_coef, k)
├── _apply_wake_effects(wind_speeds, layout, power_curve, k)
├── _apply_availability(production, maintenance_config)
├── _apply_degradation(production, degradation_rate, years)
│
└── Internal Components:
    ├── PowerCurveInterpolator  # Interpolate manufacturer power curves
    ├── WakeModel (Jensen)      # Calculate wake deficits
    └── MaintenanceModel        # Apply availability factors
```

### 3.2 Data Flow

```
Wind Speed Time Series (8760 hourly values)
    ↓
Air Density Correction (temperature, pressure, altitude)
    ↓
For each turbine in layout:
    ↓
    Calculate Wake Deficit from upstream turbines (Jensen model)
    ↓
    Apply effective wind speed = free stream - wake deficit
    ↓
    Interpolate Power from power curve
    ↓
Sum all turbine outputs → Gross hourly production
    ↓
Apply Availability (maintenance model)
    ↓
Apply Degradation (annual decay)
    ↓
Output: Net hourly production [kWh or MWh]
```

---

## 4. Mathematical Formulations

### 4.1 Wind Power Equation (IEC 61400)

Single turbine power output:

```
P = 0.5 × ρ × A × Cp × v³

Where:
- P  = Power output [W]
- ρ  = Air density [kg/m³], corrected for altitude/temperature
- A  = Rotor swept area [m²] = π × (D/2)²
- Cp = Power coefficient (from power curve, max ~0.48)
- v  = Wind speed at hub height [m/s]
```

In practice, we use the **manufacturer power curve** (P vs. v relationship) rather than calculating Cp directly.

### 4.2 Jensen Wake Model

Velocity deficit behind a turbine:

```
v_wake(x) = v_0 × [1 - (1 - √(1 - Ct)) / (1 + k × x/r₀)²]

Where:
- v_wake(x) = Wind speed at distance x downstream [m/s]
- v_0       = Free-stream wind speed [m/s]
- Ct        = Thrust coefficient (from turbine data, ~0.75-0.9)
- k         = Wake decay coefficient (0.075 onshore, 0.05 offshore)
- x         = Downstream distance [m]
- r₀        = Initial wake radius = 0.5 × D (rotor radius)
```

**Wake radius expansion:**
```
r(x) = r₀ + k × x
```

### 4.3 Multiple Wake Superposition

For turbine j receiving wakes from turbines i₁, i₂, ..., iₙ:

```
Δv_total² = Σ Δvᵢ²  (quadratic superposition)

v_effective,j = v_0 × (1 - √(Δv_total²/v_0²))
```

### 4.4 Air Density Correction

```
ρ = ρ₀ × (T₀/T) × (p/p₀)

Where:
- ρ₀ = Standard density = 1.225 kg/m³
- T₀ = Standard temperature = 288.15 K
- p₀ = Standard pressure = 101325 Pa
- T  = Actual temperature [K]
- p  = Actual pressure [Pa]
```

**Simplified altitude correction:**
```
ρ(h) = 1.225 × exp(-0.0001184 × h)

Where h = height above sea level [m]
```

### 4.5 Degradation Model

```
Production_year(y) = Production_year(0) × (1 - δ)^y

Where:
- δ = Annual degradation rate (typically 0.002-0.005 for wind)
- y = Year index (0, 1, 2, ...)
```

---

## 5. Maintenance & Availability Modeling

### 5.1 Availability Components

Total availability is the product of multiple factors:

```
A_total = A_technical × A_grid × A_environmental

Where:
- A_technical    = Turbine mechanical/electrical availability
- A_grid         = Grid availability (outages, curtailment)
- A_environmental = Weather windows (offshore access)
```

### 5.2 Maintenance Configurations

The model supports different maintenance strategies via configuration:

```python
maintenance_config = {
    'strategy': 'scheduled' | 'condition_based' | 'reactive' | 'hybrid',

    # Base availability (time-based)
    'base_availability': 0.97,  # 97% typical onshore

    # Scheduled maintenance windows
    'scheduled_maintenance': {
        'enabled': True,
        'annual_hours': 120,  # Hours per year
        'preferred_months': [5, 6, 7, 8],  # Low-wind months
        'distribution': 'concentrated' | 'spread'
    },

    # Unscheduled failures
    'failure_model': {
        'mtbf_hours': 4380,  # Mean time between failures
        'mttr_hours': 48,    # Mean time to repair
        'apply_seasonality': True  # Longer repairs in winter
    },

    # Offshore-specific
    'offshore_access': {
        'enabled': False,
        'wave_height_limit': 1.5,  # m (Hs)
        'wind_speed_limit': 12,    # m/s
        'access_window_hours': [6, 18]  # Daylight operations
    }
}
```

### 5.3 Maintenance Strategy Implementation

**Strategy 1: Simple Time-Based (Default)**
```python
hourly_availability = np.ones(n_hours) * base_availability
```

**Strategy 2: Seasonal Scheduling**
```python
# Higher availability in high-wind months (winter)
# Schedule maintenance in low-wind months (summer)
monthly_availability = {
    1: 0.98, 2: 0.98, 3: 0.97,   # Winter - high availability
    4: 0.96, 5: 0.94, 6: 0.93,   # Spring/Summer - maintenance
    7: 0.93, 8: 0.94, 9: 0.96,   # Summer/Autumn - maintenance
    10: 0.97, 11: 0.98, 12: 0.98 # Autumn/Winter - high availability
}
```

**Strategy 3: Stochastic Failures (Monte Carlo)**
```python
# Generate random failure events based on MTBF/MTTR
# Used when integrated with MC simulation framework
failure_hours = sample_exponential(mtbf)
repair_hours = sample_lognormal(mttr_mean, mttr_std)
```

### 5.4 Availability Calculation Methods

| Method | Use Case | Accuracy | Complexity |
|--------|----------|----------|------------|
| Constant factor | Quick estimates | Low | Simple |
| Monthly seasonal | Better accuracy | Medium | Medium |
| Hourly stochastic | Monte Carlo | High | Complex |

---

## 6. Input Parameters

### 6.1 Parameter Schema

```python
params = {
    'capacity': float,  # Total installed capacity [kW]

    'volume_model_params': {
        # Site parameters
        'site_type': 'onshore' | 'offshore',  # Determines wake decay default
        'hub_height': float,      # [m], default 100
        'altitude': float,        # Site elevation [m], default 0

        # Wind resource
        'wind_speed_data': np.ndarray | str,  # 8760 hourly values [m/s] or file path
        'wind_direction_data': np.ndarray | None,  # 8760 hourly values [degrees] (optional)
        'measurement_height': float,  # Height of wind data [m], default hub_height
        'wind_shear_exponent': float,  # For height correction, default 0.14

        # Turbine parameters
        'turbine_model': str | None,  # Lookup from library, or provide custom
        'rotor_diameter': float,      # [m], required if no turbine_model
        'rated_power': float,         # [kW] per turbine, required if no turbine_model
        'power_curve': dict | None,   # {wind_speed: power} pairs
        'thrust_curve': dict | None,  # {wind_speed: Ct} pairs
        'cut_in_speed': float,        # [m/s], default 3
        'cut_out_speed': float,       # [m/s], default 25
        'rated_speed': float,         # [m/s], default 12

        # Farm layout
        'n_turbines': int,            # Number of turbines
        'layout': np.ndarray | str,   # (n_turbines, 2) array of (x, y) coords [m]
                                      # or 'grid' | 'optimized' for auto-generation
        'layout_params': {            # If layout is 'grid'
            'rows': int,
            'cols': int,
            'row_spacing': float,     # Rotor diameters
            'col_spacing': float      # Rotor diameters
        },

        # Wake model
        'wake_model': 'jensen' | 'none',  # Only Jensen implemented in v1
        'wake_decay_coefficient': float | None,  # Override (default from site_type)

        # Environmental
        'air_density': float | None,  # [kg/m³], if None: calculated from altitude
        'temperature_profile': np.ndarray | None,  # 8760 hourly [K] for density correction

        # Degradation & Availability
        'degradation_rate': float,    # Annual degradation, default 0.003
        'availability': float | dict,  # Constant or maintenance_config dict
        'lifetime_years': int,         # Project lifetime, default 25
    }
}
```

### 6.2 Turbine Library (Future Extension)

```python
TURBINE_LIBRARY = {
    'vestas_v110_2000': {
        'manufacturer': 'Vestas',
        'model': 'V110-2.0',
        'rated_power_kw': 2000,
        'rotor_diameter_m': 110,
        'hub_height_m': [80, 95, 120],
        'cut_in_speed': 3.0,
        'cut_out_speed': 25.0,
        'rated_speed': 11.5,
        'power_curve': {3: 15, 4: 100, 5: 250, ...},  # Interpolated
        'thrust_curve': {3: 0.95, 4: 0.92, 5: 0.88, ...}
    },
    'siemens_sg_8.0_167': {
        'manufacturer': 'Siemens Gamesa',
        'model': 'SG 8.0-167 DD',
        'rated_power_kw': 8000,
        'rotor_diameter_m': 167,
        ...
    }
}
```

---

## 7. Output Structure

```python
# Primary output: hourly production array
hourly_production = np.ndarray  # Shape: (lifetime_years × 8760,)

# Extended output (optional, for analysis)
results = {
    'hourly_production': np.ndarray,      # kWh per hour
    'annual_production': np.ndarray,      # kWh per year
    'monthly_production': np.ndarray,     # kWh per month

    'gross_production': np.ndarray,       # Before availability losses
    'wake_losses_percent': float,         # Total wake loss %
    'availability_losses_percent': float, # Total availability loss %

    'per_turbine': {
        'annual_production': np.ndarray,  # (n_turbines, n_years)
        'capacity_factor': np.ndarray,    # (n_turbines,)
        'wake_loss': np.ndarray           # (n_turbines,) - % loss per turbine
    }
}
```

---

## 8. Implementation Plan

### Phase 1: Core Wind Power Calculation (2-3 days)

**Tasks:**
1. Create `WindVolumeModel` class skeleton implementing `VolumeModelInterface`
2. Implement power curve interpolation (linear/cubic)
3. Implement single-turbine power calculation
4. Add air density correction
5. Unit tests for single turbine

**Files:**
- `financial_model/models/wind_volume_model.py`
- `financial_model/tests/test_wind_volume_model.py`

### Phase 2: Jensen Wake Model (2 days)

**Tasks:**
1. Implement Jensen wake deficit calculation
2. Implement wake superposition (quadratic sum)
3. Add wind direction handling for layout rotation
4. Implement farm layout parsing (array or grid generation)
5. Unit tests for wake model
6. Performance benchmark (<10ms target)

**Files:**
- `financial_model/models/wind_wake_model.py` (separate module)
- `financial_model/tests/test_wind_wake_model.py`

### Phase 3: Maintenance & Availability (1-2 days)

**Tasks:**
1. Implement constant availability factor
2. Implement seasonal availability distribution
3. Add maintenance configuration parsing
4. Integration tests

**Files:**
- `financial_model/models/wind_maintenance_model.py` (optional separate module)

### Phase 4: Integration & Testing (1-2 days)

**Tasks:**
1. Full integration with FinancialModel
2. Create example wind park parameter set
3. Performance optimization (vectorization)
4. Documentation

**Files:**
- `financial_model/tests/test_integration_wind.py`

---

## 9. Performance Considerations

### 9.1 Bottlenecks

| Operation | Naive Complexity | Optimized |
|-----------|------------------|-----------|
| Wake calculation | O(n² × 8760) | Vectorized: O(n² + 8760) |
| Power curve lookup | O(8760 × interp) | Pre-compute lookup table |
| Layout rotation | O(n × 8760) | Vectorized matrix ops |

### 9.2 Optimization Strategies

**1. Vectorized Wake Calculation:**
```python
# Pre-compute pairwise distances (n_turbines × n_turbines matrix)
distances = scipy.spatial.distance.cdist(layout, layout)

# Pre-compute wake effects for all wind directions (binned)
# Only recalculate when wind direction changes significantly
```

**2. Power Curve Lookup Table:**
```python
# Create dense lookup (e.g., 0.1 m/s resolution)
power_lut = np.interp(np.arange(0, 30, 0.1), power_curve_speeds, power_curve_powers)

# Fast indexing instead of interpolation
wind_idx = (wind_speed * 10).astype(int)
power = power_lut[wind_idx]
```

**3. Direction Binning:**
```python
# Bin wind directions to 36 sectors (10° each)
# Pre-compute wake matrix for each sector
# Reduces computation by factor of ~100
```

### 9.3 Target Performance

| Scenario | Expected Time |
|----------|---------------|
| 10 turbines, 25 years | <5ms |
| 50 turbines, 25 years | <10ms |
| 100 turbines, 25 years | <20ms |

---

## 10. Validation & Testing

### 10.1 Unit Tests

```python
def test_power_curve_interpolation():
    """Verify power output matches curve at known points."""

def test_jensen_wake_deficit():
    """Verify wake velocity matches analytical solution."""

def test_wake_superposition():
    """Multiple upstream turbines combine correctly."""

def test_air_density_correction():
    """Production scales with density."""

def test_wind_shear_correction():
    """Hub height extrapolation is correct."""
```

### 10.2 Integration Tests

```python
def test_wind_farm_10_turbines():
    """Full calculation for 10-turbine onshore farm."""
    params = {...}
    model = WindVolumeModel()
    production = model.calculate(params)

    assert production.shape == (25 * 8760,)
    assert 0 < production.sum() / (10 * 3000 * 8760) < 0.5  # CF < 50%

def test_offshore_vs_onshore():
    """Offshore has higher wake losses."""
    # Same layout, different k coefficient
    # Verify offshore production < onshore production
```

### 10.3 Benchmark Tests

```python
def test_performance_under_10ms():
    """Single evaluation must complete in <10ms."""
    import time

    params = create_50_turbine_params()
    model = WindVolumeModel()

    # Warm-up
    model.calculate(params)

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        model.calculate(params)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / 100) * 1000
    assert avg_ms < 10, f"Average time {avg_ms}ms exceeds 10ms target"
```

---

## 11. Example: Onshore Wind Farm

### 11.1 Parameter Set

```python
onshore_wind_params = {
    'capacity': 30000,  # 30 MW (10 × 3 MW turbines)

    'volume_model_params': {
        'site_type': 'onshore',
        'hub_height': 100,
        'altitude': 200,  # 200m above sea level

        # Wind resource (would come from measurement or reanalysis data)
        'wind_speed_data': np.random.weibull(2.2, 8760) * 7,  # ~7 m/s avg
        'wind_direction_data': np.random.uniform(180, 270, 8760),  # Prevailing SW

        # Turbine
        'rotor_diameter': 120,
        'rated_power': 3000,  # 3 MW per turbine
        'cut_in_speed': 3.0,
        'cut_out_speed': 25.0,
        'rated_speed': 12.0,
        'power_curve': {
            3: 30, 4: 150, 5: 350, 6: 600, 7: 950, 8: 1400,
            9: 1900, 10: 2400, 11: 2800, 12: 3000, 13: 3000,
            14: 3000, 15: 3000, 16: 3000, 17: 3000, 18: 3000,
            19: 3000, 20: 3000, 21: 3000, 22: 3000, 23: 3000,
            24: 3000, 25: 3000
        },
        'thrust_curve': {
            3: 0.95, 4: 0.92, 5: 0.88, 6: 0.85, 7: 0.82,
            8: 0.79, 9: 0.75, 10: 0.70, 11: 0.60, 12: 0.50,
            13: 0.40, 14: 0.35, 15: 0.30, 16: 0.25, 17: 0.20,
            18: 0.18, 19: 0.16, 20: 0.14, 21: 0.12, 22: 0.10,
            23: 0.09, 24: 0.08, 25: 0.07
        },

        # Layout: 2 rows × 5 columns
        'n_turbines': 10,
        'layout': 'grid',
        'layout_params': {
            'rows': 2,
            'cols': 5,
            'row_spacing': 7,   # 7D = 840m
            'col_spacing': 5    # 5D = 600m
        },

        # Wake model
        'wake_model': 'jensen',
        'wake_decay_coefficient': None,  # Use default for onshore (0.075)

        # Availability & Degradation
        'degradation_rate': 0.003,
        'availability': 0.97,
        'lifetime_years': 25
    }
}
```

### 11.2 Expected Results

For a well-sited 30 MW onshore wind farm with ~7 m/s average wind speed:

```python
expected = {
    'gross_capacity_factor': 0.32,      # Before wake losses
    'wake_losses_percent': 8.0,         # ~8% with good spacing
    'availability_losses_percent': 3.0, # 97% availability
    'net_capacity_factor': 0.28,        # After all losses

    'annual_production_mwh': 73_500,    # ~30 MW × 8760h × 0.28
    'lcoe_target': 0.045                # €/kWh (competitive onshore)
}
```

---

## 12. Open Questions & Future Work

### 12.1 Not Included in V1.0

1. **External Wind Data APIs** (e.g., ERA5, MERRA-2 reanalysis)
2. **Advanced Wake Models** (Bastankhah-Porté-Agel, Gaussian)
3. **Turbine Library** (pre-defined manufacturer curves)
4. **Yaw Misalignment** modeling
5. **Icing Losses** (cold climate)
6. **Noise Curtailment** (operational restrictions)
7. **Shadow Flicker** curtailment

### 12.2 Integration Questions

- **How to handle wind data input?**
  - V1: User provides array or file path
  - Future: API integration for automatic fetch

- **Layout optimization?**
  - V1: User provides layout or simple grid
  - Future: Genetic algorithm optimization

---

## 13. References

### Academic Sources
- Jensen, N.O. (1983). "A Note on Wind Generator Interaction." Risø-M-2411.
- Katic, I. et al. (1986). "A Simple Model for Cluster Efficiency." EWEC'86.
- [WES Copernicus - Jensen Parameterization](https://wes.copernicus.org/articles/7/2407/2022/)
- [ScienceDirect - Wake Model Review](https://www.sciencedirect.com/science/article/abs/pii/S030626191830802X)

### Industry Standards
- [IEC 61400-12-1:2022](https://webstore.iec.ch/en/publication/68499) - Power Performance Measurements
- [IEC 61400-26-1](https://standards.globalspec.com/std/13327126/iec-61400-26-1) - Availability Definitions
- [DNV GL Availability White Paper](https://www.ourenergypolicy.org/wp-content/uploads/2017/08/Definitions-of-availability-terms-for-the-wind-industry-white-paper-09-08-2017.pdf)

### Research Projects
- [EuroWindWakes (2025)](https://www.offshorewind.biz/2025/05/26/new-research-project-aims-to-create-more-accurate-wake-modelling-for-offshore-wind/)

---

## 14. Approval Checklist

Before implementation begins, confirm:

- [ ] Unified onshore/offshore model approach approved
- [ ] Jensen wake model acceptable for V1.0
- [ ] Maintenance configuration schema approved
- [ ] Performance target (<10ms) confirmed
- [ ] Parameter schema compatible with existing financial model
- [ ] Test coverage requirements agreed

---

**Document Status:** Ready for Review
**Next Step:** Approval → Begin Phase 1 Implementation
