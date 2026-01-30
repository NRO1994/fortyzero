# Financial Model Specification
## Energy Infrastructure Investment Analysis Tool

**Version:** 1.0  
**Date:** 2026-01-30  
**Purpose:** Generic, high-performance financial model for energy infrastructure projects (PV, Wind, Heat Networks, BESS, etc.)

---

## 1. Overview

### 1.1 Objective
Create a robust, fast (<10ms per evaluation), and versatile financial model that:
- Calculates NPV, IRR (project & equity), Payback Period, DSCR, and LCOE
- Supports multiple energy asset types through modular architecture
- Integrates with Monte Carlo simulation framework
- Handles monthly operational granularity with annual DCF aggregation

### 1.2 Key Design Principles
- **Modularity:** Swappable components (revenue models, cost models, asset templates)
- **Performance:** NumPy vectorization, target <10ms evaluation time
- **Genericity:** Applicable across PV, Wind, Heat Networks, CHP, and other energy assets
- **Interfaces:** Clear contracts between components for extensibility

---

## 2. Architecture

### 2.1 Component Structure

```
FinancialModel
├── VolumeModel (Interface)
│   └── Returns: monthly energy/volume production [kWh, MWh, or units]
├── RevenueModel
│   ├── Uses: VolumeModel output × Price streams
│   └── Returns: monthly revenues by stream
├── CostModel
│   ├── CAPEX (development, construction, replacement)
│   ├── OPEX (fixed, variable)
│   └── Returns: monthly costs by category
├── TaxModel
│   ├── Uses: Asset template for tax rates & depreciation
│   └── Returns: annual tax liability
├── FinancingModel
│   ├── Debt structure (annuity loan)
│   └── Returns: interest & principal payments
├── DCFEngine
│   ├── Aggregates monthly → annual cash flows
│   ├── Calculates NPV, IRR_project, IRR_equity
│   └── Calculates DSCR, Payback, LCOE
└── AssetTemplateLibrary
    └── Asset-specific parameters (depreciation, tax, lifetime)
```

### 2.2 Data Flow

```
Input Parameters (structured dict)
    ↓
VolumeModel.calculate() → [monthly volumes]
    ↓
RevenueModel.calculate(volumes, prices) → [monthly revenues]
    ↓
CostModel.calculate() → [monthly costs]
    ↓
Aggregate to annual cash flows
    ↓
TaxModel.calculate(annual_ebitda) → [annual taxes]
    ↓
FinancingModel.calculate() → [annual debt service]
    ↓
DCFEngine.calculate() → {NPV, IRR_project, IRR_equity, DSCR, Payback, LCOE}
```

---

## 3. Temporal Structure

### 3.1 Time Periods

**Development Period:**
- Pre-construction phase
- Low development costs (planning, permits, engineering)
- Duration: user-defined (typically -1 to 0 years)

**Construction Period:**
- Asset installation and grid connection
- Phased CAPEX deployment
- Duration: user-defined (typically 0 to 0.5 years)

**Operational Period:**
- Monthly granularity for operations (revenues, OPEX)
- Annual aggregation for DCF calculations
- Duration: asset-specific (e.g., PV: 25 years, Heat Network: 40 years)

### 3.2 Time Indexing

```python
# Month indexing: 0 to (project_lifetime_years × 12) - 1
# Year indexing: 0 to project_lifetime_years - 1

# Example for 25-year PV project:
# Development: Year -1 (12 months before construction)
# Construction: Year 0, Months 0-5
# Operation: Year 0, Month 6 → Year 24, Month 11
```

### 3.3 Discounting Convention
- **End-of-period cash flows**
- Annual discounting: `PV = CF_t / (1 + WACC)^t` where t = 1, 2, 3, ...
- Monthly cash flows aggregated to annual before discounting

---

## 4. Input Structure

### 4.1 Parameter Dictionary (Structured)

```python
params = {
    'project': {
        'name': str,
        'asset_type': str,  # 'pv', 'wind', 'heat_network', 'chp'
        'lifetime_years': int,
        'development_duration_years': float,
        'construction_duration_years': float,
        'start_date': str  # ISO format, for reporting only
    },
    
    'technical': {
        'capacity': float,  # kW, MW, or relevant unit
        'volume_model_params': dict  # Passed to VolumeModel interface
        # VolumeModel handles degradation, availability, seasonality
    },
    
    'financial': {
        'capex': {
            'development': [
                {'phase': 'development', 'year': -1, 'month': 0, 'amount': float, 'category': str},
                # e.g., {'phase': 'development', 'year': -1, 'month': 0, 'amount': 50000, 'category': 'permits'}
            ],
            'construction': [
                {'phase': 'construction', 'year': 0, 'month': 0, 'amount': float, 'category': str},
                {'phase': 'construction', 'year': 0, 'month': 3, 'amount': float, 'category': str},
                # e.g., {'phase': 'construction', 'year': 0, 'month': 0, 'amount': 500000, 'category': 'modules'}
            ],
            'replacement': [
                {'phase': 'replacement', 'year': 10, 'month': 0, 'amount': float, 'category': str},
                # e.g., {'phase': 'replacement', 'year': 10, 'month': 0, 'amount': 50000, 'category': 'inverters'}
            ],
            'decommissioning': {
                'enabled': bool,
                'year': int,  # typically = lifetime_years
                'amount': float,
                'reserve_annual': bool  # if True, annual reserves; if False, one-time cost
            }
        },
        
        'opex': {
            'fixed': [
                {
                    'category': str,  # e.g., 'maintenance', 'insurance', 'lease'
                    'annual_amount': float,
                    'indexed': bool,  # if True, escalate with inflation
                    'escalation_rate': float  # custom rate, or use inflation if null
                },
            ],
            'variable': [
                {
                    'category': str,  # e.g., 'fuel', 'electricity_purchase'
                    'unit_cost': float,  # per kWh, per unit
                    'indexed': bool,
                    'escalation_rate': float
                }
                # Volume comes from VolumeModel or separate driver
            ]
        },
        
        'revenue': {
            'streams': [
                {
                    'name': str,  # e.g., 'eeg_tariff', 'merchant_sales', 'heat_sales'
                    'type': str,  # 'fixed_price', 'market_based', 'regulated'
                    'price_structure': {
                        'fixed_period': {
                            'start_year': int,
                            'end_year': int,
                            'price': float,  # €/kWh or €/MWh
                            'indexed': bool,
                            'escalation_rate': float
                        },
                        'market_period': {
                            'start_year': int,
                            'end_year': int,
                            'price_curve': str,  # 'high', 'mid', 'low' → links to PriceCurveInterface
                        }
                    }
                }
            ]
        },
        
        'financing': {
            'equity_share': float,  # 0.0 to 1.0
            'debt': {
                'principal': float,  # total debt amount
                'interest_rate': float,  # annual, fixed
                'term_years': int,  # loan duration
                'type': 'annuity'  # only annuity for now
            }
        },
        
        'tax': {
            'corporate_tax_rate': float,  # effective rate (default from asset template)
            'depreciation_method': 'linear',  # only linear for now
            'depreciation_years': int,  # from asset template or override
        },
        
        'discount': {
            'wacc': float,  # Weighted Average Cost of Capital
            # or calculated from:
            'cost_of_equity': float,
            'cost_of_debt': float,
            'tax_rate': float  # for debt tax shield
        },
        
        'inflation': {
            'base_rate': float,  # applied where indexed=True and no custom rate
            # Future: could be time series per year
        }
    }
}
```

---

## 5. Interfaces

### 5.1 VolumeModel Interface

**Purpose:** Calculate monthly energy/volume production for the asset

```python
class VolumeModelInterface:
    """
    Abstract interface for volume/energy production models.
    Implementations handle asset-specific physics, degradation, availability.
    """
    
    def calculate(self, params: dict) -> np.ndarray:
        """
        Calculate monthly production volumes.
        
        Args:
            params: technical parameters from input dict
            
        Returns:
            np.ndarray of shape (n_months,) with production in kWh, MWh, or relevant unit
            
        Example for PV:
            - Inputs: irradiance data, capacity, degradation_rate, availability
            - Calculation: irradiance × capacity × (1 - degradation)^year × availability
            - Output: [month_0_kwh, month_1_kwh, ..., month_N_kwh]
        
        Example for Heat Network:
            - Inputs: customer_growth_curve, consumption_per_customer, seasonality
            - Calculation: customers_t × consumption × seasonal_factor_t
            - Output: [month_0_mwh, month_1_mwh, ..., month_N_mwh]
        """
        raise NotImplementedError
```

**Implementation Notes:**
- VolumeModel is **not** part of the core FinancialModel
- Developed separately and passed as dependency injection
- FinancialModel receives pre-calculated monthly volumes
- Allows for complex models (e.g., weather simulations, customer growth S-curves) without cluttering financial logic

### 5.2 PriceCurve Interface

**Purpose:** Provide electricity/commodity prices for merchant periods

```python
class PriceCurveInterface:
    """
    Abstract interface for market price curves.
    Handles loading and interpolation of price data.
    """
    
    def __init__(self, curve_type: str):
        """
        Initialize with curve type.
        
        Args:
            curve_type: 'high', 'mid', 'low' (selected by MC framework)
        """
        self.curve_type = curve_type
        
    def get_monthly_prices(self, start_year: int, end_year: int) -> np.ndarray:
        """
        Get monthly average prices for specified period.
        
        Args:
            start_year: first year of merchant period
            end_year: last year of merchant period
            
        Returns:
            np.ndarray of shape (n_months,) with prices in €/kWh or €/MWh
            
        Example:
            - CSV contains hourly prices for 30 years
            - Method aggregates to monthly averages
            - Returns array aligned with project timeline
        """
        raise NotImplementedError
```

**Implementation Notes:**
- PriceCurveInterface handles CSV parsing externally
- FinancialModel receives instantiated price curve object
- MC framework instantiates with selected scenario ('high'/'mid'/'low')

---

## 6. Asset Template Library

### 6.1 Structure

```python
ASSET_TEMPLATES = {
    'pv': {
        'depreciation_years': 20,
        'corporate_tax_rate': 0.30,  # Germany: ~30% effective (corp + trade + solidarity)
        'default_lifetime_years': 25,
        'typical_wacc': 0.04,
        'decommissioning_cost_per_kw': 50  # €/kW, optional default
    },
    
    'wind': {
        'depreciation_years': 20,
        'corporate_tax_rate': 0.30,
        'default_lifetime_years': 25,
        'typical_wacc': 0.05,
        'decommissioning_cost_per_kw': 100
    },
    
    'heat_network': {
        'depreciation_years': 40,
        'corporate_tax_rate': 0.30,
        'default_lifetime_years': 40,
        'typical_wacc': 0.06,
        'decommissioning_cost_per_kw': 20
    },
    
    'chp': {
        'depreciation_years': 15,
        'corporate_tax_rate': 0.30,
        'default_lifetime_years': 20,
        'typical_wacc': 0.06,
        'decommissioning_cost_per_kw': 75
    }
}
```

### 6.2 Usage
- User specifies `asset_type` in parameters
- Template provides defaults for depreciation, tax rates, typical WACC
- User can override any template value in input parameters
- Single library for both financial and tax parameters (simpler than separate libraries)

---

## 7. Calculation Methodology

### 7.1 Revenue Calculation

**Monthly Level:**
```python
for stream in revenue_streams:
    for month in range(n_months):
        year = month // 12
        
        # Determine price
        if in_fixed_period(year, stream):
            price = stream.fixed_price × (1 + escalation)^year
        elif in_market_period(year, stream):
            price = price_curve.get_price(year, month)
        
        # Revenue = Volume × Price
        revenue[stream][month] = volume[month] × price
```

**Annual Aggregation:**
```python
annual_revenue[year] = sum(monthly_revenue[year×12 : (year+1)×12])
```

### 7.2 Cost Calculation

**CAPEX:**
```python
# Development
capex_development = sum of all development phase costs

# Construction
capex_construction = sum of all construction phase costs

# Replacement
capex_replacement[year] = sum of all replacements in that year

# Decommissioning
if decommissioning.enabled:
    if decommissioning.reserve_annual:
        annual_reserve = decommissioning.amount / project_lifetime
        opex_fixed[year] += annual_reserve
    else:
        capex_decommissioning[final_year] = decommissioning.amount
```

**OPEX Fixed:**
```python
for category in opex_fixed:
    for year in range(n_years):
        if category.indexed:
            escalation = category.escalation_rate or inflation.base_rate
            opex[category][year] = category.annual_amount × (1 + escalation)^year
        else:
            opex[category][year] = category.annual_amount
```

**OPEX Variable:**
```python
for category in opex_variable:
    for month in range(n_months):
        year = month // 12
        
        # Volume driver (from VolumeModel or specific driver)
        volume_driver = get_volume_driver(category)
        
        # Unit cost escalation
        if category.indexed:
            escalation = category.escalation_rate or inflation.base_rate
            unit_cost = category.unit_cost × (1 + escalation)^year
        else:
            unit_cost = category.unit_cost
        
        opex_var[category][month] = volume_driver[month] × unit_cost

# Aggregate to annual
annual_opex_var[year] = sum(monthly_opex_var[year×12 : (year+1)×12])
```

### 7.3 Tax Calculation

**Depreciation:**
```python
# Linear depreciation
annual_depreciation = total_depreciable_capex / depreciation_years

# Depreciable CAPEX = development + construction + replacement
# Typically excludes land, includes installation
```

**Taxable Income:**
```python
for year in range(n_years):
    ebitda[year] = revenue[year] - opex_fixed[year] - opex_variable[year]
    ebit[year] = ebitda[year] - depreciation[year]
    
    # Interest expense (if debt-financed)
    interest[year] = debt_balance[year] × interest_rate
    
    ebt[year] = ebit[year] - interest[year]
    
    # Tax liability
    if ebt[year] > 0:
        tax[year] = ebt[year] × corporate_tax_rate
    else:
        # Loss carry-forward (simplified: assume infinite carry-forward)
        accumulated_losses += abs(ebt[year])
        tax[year] = 0
    
    # Deduct carried losses in future profitable years
    if ebt[year] > 0 and accumulated_losses > 0:
        offset = min(ebt[year], accumulated_losses)
        tax[year] = (ebt[year] - offset) × corporate_tax_rate
        accumulated_losses -= offset
```

### 7.4 Financing Calculation

**Annuity Loan:**
```python
# Annuity payment (constant annual payment)
annuity = principal × (interest_rate × (1 + interest_rate)^term) / ((1 + interest_rate)^term - 1)

for year in range(term_years):
    interest_payment[year] = debt_balance[year] × interest_rate
    principal_payment[year] = annuity - interest_payment[year]
    debt_balance[year+1] = debt_balance[year] - principal_payment[year]
```

### 7.5 Cash Flow Calculation

**Unlevered Free Cash Flow (Project):**
```python
for year in range(n_years):
    fcf_unlevered[year] = (
        ebitda[year]
        - capex[year]
        - tax_on_ebit[year]  # Tax calculated without interest deduction
    )
```

**Levered Free Cash Flow (Equity):**
```python
for year in range(n_years):
    fcf_levered[year] = (
        ebitda[year]
        - capex[year]
        - tax[year]  # Tax with interest deduction
        - interest_payment[year]
        - principal_payment[year]
    )

# Initial equity investment
fcf_levered[0] -= equity_amount
fcf_levered[0] += debt_drawdown  # positive inflow from debt
```

### 7.6 NPV Calculation

**Project NPV (Unlevered):**
```python
npv_project = sum([fcf_unlevered[t] / (1 + wacc)^t for t in range(1, n_years+1)])
npv_project -= initial_investment  # CAPEX at t=0 not discounted
```

**Equity NPV (Levered):**
```python
npv_equity = sum([fcf_levered[t] / (1 + cost_of_equity)^t for t in range(1, n_years+1)])
```

### 7.7 IRR Calculation

**Project IRR:**
```python
# Include all unlevered cash flows including initial CAPEX
cash_flows_project = [-initial_investment] + list(fcf_unlevered[1:])
irr_project = scipy.optimize.newton(lambda r: npv(cash_flows_project, r), x0=0.1)
```

**Equity IRR:**
```python
# Include equity investment and all levered cash flows
cash_flows_equity = [-equity_amount] + list(fcf_levered[1:])
irr_equity = scipy.optimize.newton(lambda r: npv(cash_flows_equity, r), x0=0.1)
```

### 7.8 DSCR Calculation

```python
# Debt Service Coverage Ratio
# DSCR = Cash Available for Debt Service / Debt Service

for year in range(term_years):
    cash_available = ebitda[year] - tax[year] - capex[year]  # Operating CF before debt
    debt_service = interest_payment[year] + principal_payment[year]
    
    dscr[year] = cash_available / debt_service
    
# Typical bank requirement: min DSCR ≥ 1.2-1.3
min_dscr = min(dscr)
avg_dscr = mean(dscr)
```

### 7.9 Payback Period

```python
# Simple Payback (undiscounted)
cumulative_cf = cumsum(fcf_levered)  # or fcf_unlevered
payback_simple = first year where cumulative_cf > 0

# Discounted Payback
cumulative_discounted_cf = cumsum([fcf[t] / (1 + wacc)^t for t in range(n_years)])
payback_discounted = first year where cumulative_discounted_cf > 0
```

### 7.10 LCOE Calculation

```python
# Levelized Cost of Energy
# LCOE = NPV(Total Costs) / NPV(Total Energy Production)

total_costs = capex + opex_fixed + opex_variable + tax + interest

npv_costs = sum([total_costs[t] / (1 + wacc)^t for t in range(n_years)])
npv_energy = sum([total_energy_production[t] / (1 + wacc)^t for t in range(n_years)])

lcoe = npv_costs / npv_energy  # €/kWh or €/MWh
```

**Note:** Total energy production comes from VolumeModel (already accounts for degradation, availability)

---

## 8. Output Structure

### 8.1 Primary KPIs

```python
results = {
    'kpis': {
        'npv_project': float,  # € (unlevered)
        'npv_equity': float,   # € (levered)
        'irr_project': float,  # % (unlevered)
        'irr_equity': float,   # % (levered)
        'payback_simple': float,  # years
        'payback_discounted': float,  # years
        'dscr_min': float,
        'dscr_avg': float,
        'lcoe': float  # €/kWh or €/MWh
    }
}
```

### 8.2 Detailed Cash Flows (Optional for Debugging/Analysis)

```python
results = {
    'kpis': {...},
    
    'cash_flows': {
        'annual': {
            'revenue': np.ndarray,  # shape (n_years,)
            'opex_fixed': np.ndarray,
            'opex_variable': np.ndarray,
            'capex': np.ndarray,
            'depreciation': np.ndarray,
            'tax': np.ndarray,
            'interest': np.ndarray,
            'principal': np.ndarray,
            'fcf_unlevered': np.ndarray,
            'fcf_levered': np.ndarray
        },
        'monthly': {
            'revenue': np.ndarray,  # shape (n_months,)
            'opex_variable': np.ndarray,
            'volume': np.ndarray
        }
    },
    
    'financing': {
        'debt_balance': np.ndarray,  # shape (term_years,)
        'dscr': np.ndarray
    }
}
```

---

## 9. Performance Requirements

### 9.1 Target Metrics
- **Single evaluation time:** <10ms (without parallelization)
- **Monte Carlo throughput:** >100 evaluations/second
- **Memory efficiency:** <50 MB per evaluation

### 9.2 Optimization Strategies

**NumPy Vectorization:**
```python
# BAD: Python loops
for year in range(n_years):
    revenue[year] = volume[year] * price[year]

# GOOD: NumPy operations
revenue = volume * price  # element-wise multiplication
```

**Pre-allocation:**
```python
# Pre-allocate arrays
n_years = project_lifetime_years
n_months = n_years * 12

revenue_monthly = np.zeros(n_months)
opex_monthly = np.zeros(n_months)
fcf_annual = np.zeros(n_years)
```

**Avoid Repeated Calculations:**
```python
# Calculate discount factors once
discount_factors = 1 / (1 + wacc) ** np.arange(1, n_years + 1)

# Apply in one operation
npv = np.sum(cash_flows * discount_factors)
```

**Minimize Object Creation:**
- Reuse arrays where possible
- Avoid creating intermediate dicts/lists in hot loops

---

## 10. Validation & Error Handling

### 10.1 Assumptions
- Error handling is managed by **overarching MC framework**
- FinancialModel focuses on mathematical correctness
- Invalid inputs caught upstream

### 10.2 Internal Checks
- **IRR convergence:** If Newton-Raphson fails, return `np.nan` (not exception)
- **Division by zero:** Check denominators (e.g., DSCR calculation)
- **Negative square roots:** In financial calcs (shouldn't occur with valid inputs)

### 10.3 Edge Cases
```python
# Example: No debt financing
if equity_share == 1.0:
    # Skip all debt calculations
    irr_equity = irr_project
    dscr = np.inf  # or np.nan

# Example: Zero production
if np.sum(volume) == 0:
    lcoe = np.inf

# Example: Immediate payback
if all(fcf > 0):
    payback = 0
```

---

## 11. Extensibility & Future Enhancements

### 11.1 Planned Extensions (Not in V1.0)

**Multi-period inflation:**
```python
# Currently: single inflation rate
'inflation': {'base_rate': 0.02}

# Future: time series
'inflation': {'rates': [0.02, 0.025, 0.03, ...]}  # year-by-year
```

**Tax loss carry-back:**
```python
# Currently: infinite carry-forward only
# Future: configurable carry-back (1-2 years) and limited carry-forward (Germany: unlimited)
```

**Working Capital:**
```python
'working_capital': {
    'receivables_days': 30,
    'payables_days': 60,
    'initial_wc': float
}
```

**Advanced debt structures:**
- Interest-only periods
- Balloon payments
- Variable interest rates (EURIBOR + margin)

**Currency handling:**
- Multi-currency projects
- FX hedging costs

### 11.2 Extension Points

**New VolumeModels:**
```python
# Easy to add new asset types
class BatteryStorageVolumeModel(VolumeModelInterface):
    def calculate(self, params):
        # Implement charge/discharge cycles, degradation
        return monthly_throughput
```

**New Revenue Structures:**
```python
# E.g., capacity markets, ancillary services
'revenue': {
    'streams': [
        {'name': 'capacity_payment', 'type': 'capacity_market', ...},
        {'name': 'frequency_regulation', 'type': 'ancillary', ...}
    ]
}
```

**Advanced Tax Models:**
```python
class TaxModelAdvanced(TaxModelInterface):
    def calculate(self, ebit, interest, params):
        # Implement complex tax rules
        # - Minimum taxation
        # - Investment tax credits
        # - Accelerated depreciation
        return tax_liability
```

---

## 12. Implementation Guidance for Claude Code

### 12.1 File Structure

```
financial_model/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── financial_model.py      # Main FinancialModel class
│   ├── dcf_engine.py            # NPV, IRR calculations
│   └── kpi_calculator.py        # DSCR, Payback, LCOE
├── models/
│   ├── __init__.py
│   ├── revenue_model.py
│   ├── cost_model.py
│   ├── tax_model.py
│   └── financing_model.py
├── interfaces/
│   ├── __init__.py
│   ├── volume_model.py          # Abstract interface
│   └── price_curve.py           # Abstract interface
├── templates/
│   ├── __init__.py
│   └── asset_templates.py       # ASSET_TEMPLATES dict
├── utils/
│   ├── __init__.py
│   └── financial_utils.py       # Helper functions (annuity calc, etc.)
└── tests/
    ├── test_financial_model.py
    ├── test_revenue_model.py
    └── test_pv_case.py          # Integration test for PV-EEG case
```

### 12.2 Dependencies

**Required:**
- `numpy` >= 1.24
- `scipy` >= 1.10 (for IRR optimization)

**Optional (for testing):**
- `pytest`
- `pandas` (for easier test data manipulation)

### 12.3 Code Style
- **Docstrings:** Google style
- **Type hints:** Mandatory for public methods
- **Naming:** 
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

### 12.4 Testing Strategy

**Unit Tests:**
- Each model component (Revenue, Cost, Tax, Financing) independently
- Edge cases (zero debt, zero production, negative cash flows)

**Integration Test (PV-EEG Case):**
```python
def test_pv_eeg_base_case():
    """
    Test complete PV project with EEG tariff + merchant tail.
    
    Expected results (approximate):
    - NPV_project: ~€200k
    - IRR_project: ~7%
    - LCOE: ~€0.065/kWh
    - Min DSCR: >1.3
    """
    params = {...}  # Full PV parameter set
    model = FinancialModel(asset_type='pv')
    results = model.calculate(params)
    
    assert results['kpis']['npv_project'] > 0
    assert 0.05 < results['kpis']['irr_project'] < 0.10
    assert results['kpis']['dscr_min'] > 1.2
```

**Performance Test:**
```python
def test_performance_10ms():
    """Ensure single evaluation <10ms."""
    params = {...}
    model = FinancialModel(asset_type='pv')
    
    start = time.time()
    for _ in range(100):
        model.calculate(params)
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / 100) * 1000
    assert avg_time_ms < 10
```

### 12.5 Documentation

**README.md:**
- Quick start example (PV case)
- Parameter reference
- Example usage with MC framework

**API Reference:**
- Auto-generated from docstrings
- Interface contracts clearly documented

---

## 13. Example: PV-EEG Project

### 13.1 Parameter Set

```python
pv_eeg_params = {
    'project': {
        'name': 'Solar Park Musterstadt',
        'asset_type': 'pv',
        'lifetime_years': 25,
        'development_duration_years': 1,
        'construction_duration_years': 0.5,
        'start_date': '2026-07-01'
    },
    
    'technical': {
        'capacity': 1000,  # kW
        'volume_model_params': {
            # Passed to PVVolumeModel (to be implemented separately)
            'irradiance_profile': 'typical_year_germany',
            'degradation_rate': 0.005,  # 0.5% per year
            'availability': 0.97,
            'module_efficiency': 0.20
        }
    },
    
    'financial': {
        'capex': {
            'development': [
                {'phase': 'development', 'year': -1, 'month': 0, 'amount': 30000, 'category': 'permits_engineering'}
            ],
            'construction': [
                {'phase': 'construction', 'year': 0, 'month': 0, 'amount': 400000, 'category': 'modules'},
                {'phase': 'construction', 'year': 0, 'month': 2, 'amount': 150000, 'category': 'inverters'},
                {'phase': 'construction', 'year': 0, 'month': 4, 'amount': 200000, 'category': 'installation'},
                {'phase': 'construction', 'year': 0, 'month': 5, 'amount': 50000, 'category': 'grid_connection'}
            ],
            'replacement': [
                {'phase': 'replacement', 'year': 10, 'month': 0, 'amount': 80000, 'category': 'inverters'},
                {'phase': 'replacement', 'year': 20, 'month': 0, 'amount': 80000, 'category': 'inverters'}
            ],
            'decommissioning': {
                'enabled': True,
                'year': 25,
                'amount': 50000,
                'reserve_annual': True
            }
        },
        
        'opex': {
            'fixed': [
                {'category': 'maintenance', 'annual_amount': 12000, 'indexed': True, 'escalation_rate': None},
                {'category': 'insurance', 'annual_amount': 3000, 'indexed': True, 'escalation_rate': None},
                {'category': 'land_lease', 'annual_amount': 5000, 'indexed': True, 'escalation_rate': 0.02}
            ],
            'variable': []  # PV has no variable OPEX typically
        },
        
        'revenue': {
            'streams': [
                {
                    'name': 'eeg_tariff',
                    'type': 'fixed_price',
                    'price_structure': {
                        'fixed_period': {
                            'start_year': 0,
                            'end_year': 19,  # 20 years EEG
                            'price': 0.073,  # €/kWh (bid price)
                            'indexed': False,
                            'escalation_rate': 0.0
                        },
                        'market_period': {
                            'start_year': 20,
                            'end_year': 24,
                            'price_curve': 'mid'  # links to PriceCurveInterface
                        }
                    }
                }
            ]
        },
        
        'financing': {
            'equity_share': 0.25,  # 25% equity, 75% debt
            'debt': {
                'principal': 600000,  # 75% of ~800k total CAPEX
                'interest_rate': 0.045,
                'term_years': 15,
                'type': 'annuity'
            }
        },
        
        'tax': {
            'corporate_tax_rate': 0.30,
            'depreciation_method': 'linear',
            'depreciation_years': 20
        },
        
        'discount': {
            'wacc': 0.055,
            'cost_of_equity': 0.08,
            'cost_of_debt': 0.045,
            'tax_rate': 0.30
        },
        
        'inflation': {
            'base_rate': 0.02
        }
    }
}
```

### 13.2 Expected Results (Approximate)

With bid price of €0.073/kWh:

```python
expected_results = {
    'kpis': {
        'npv_project': 180000,  # €180k (unlevered)
        'npv_equity': 120000,   # €120k (levered)
        'irr_project': 0.068,   # 6.8%
        'irr_equity': 0.095,    # 9.5% (leverage effect)
        'payback_simple': 12.5,
        'payback_discounted': 16.2,
        'dscr_min': 1.35,
        'dscr_avg': 1.58,
        'lcoe': 0.0625  # €0.0625/kWh
    }
}
```

---

## 14. Summary & Next Steps

### 14.1 Specification Completeness

This specification covers:
- Temporal structure (monthly operations, annual DCF)  
- Modular architecture with clear interfaces  
- Complete calculation methodology (Revenue, Costs, Tax, Financing, KPIs)  
- Asset templates for multiple energy types  
- Performance targets and optimization strategies  
- Extensibility for future enhancements  
- Complete PV-EEG example case  

### 14.2 Implementation Checklist

1. **Core Components**
   - [ ] FinancialModel main class
   - [ ] DCFEngine (NPV, IRR calculations)
   - [ ] KPICalculator (DSCR, Payback, LCOE)

2. **Sub-Models**
   - [ ] RevenueModel
   - [ ] CostModel
   - [ ] TaxModel
   - [ ] FinancingModel

3. **Interfaces**
   - [ ] VolumeModelInterface (abstract)
   - [ ] PriceCurveInterface (abstract)

4. **Templates & Utils**
   - [ ] AssetTemplates library
   - [ ] Financial utility functions

5. **Testing**
   - [ ] Unit tests for each component
   - [ ] PV-EEG integration test
   - [ ] Performance benchmark (<10ms)

6. **Documentation**
   - [ ] README with quickstart
   - [ ] API reference
   - [ ] Example notebooks

### 14.3 Ready for Claude Code

This specification is **complete and ready** for implementation by Claude Code. Key features:
- Unambiguous technical requirements
- Complete mathematical formulations
- Clear interfaces for modularity
- Concrete example (PV-EEG) with expected results
- Performance targets and optimization guidance

**Recommended Implementation Order:**
1. Core financial utilities (annuity, NPV, IRR)
2. Asset templates
3. Individual models (Revenue, Cost, Tax, Financing)
4. DCF engine and KPI calculator
5. Main FinancialModel orchestrator
6. Interfaces (abstract classes)
7. Tests and validation
8. Example PV case

## 15. Documentation Requirements
### 15.1 Code Documentation

All classes and methods with docstrings (Google style)
Type hints for all function signatures
Inline comments for complex financial logic

### 15.2 User Documentation

Usage examples for each asset type
Parameter descriptions with units
Common pitfalls and how to avoid them

### 15.3 Developer Documentation

Architecture diagram (component interactions)
Data flow diagram (inputs → calculations → outputs)
Extension guide for new asset types


## 16. Acceptance Criteria
### 16.1 Functional
- Calculates all required KPIs correctly
- Handles all specified asset types (PV, Wind, Heat, KWK)
- Interfaces work as specified (Volume, Price Curve)
- Template system functional
### 16.2 Non-Functional
- Performance: <10ms per calculation
- Numerical stability: No overflow/underflow errors
- Deterministic: Same inputs → Same outputs
- Readable: Code can be understood by team members
### 16.3 Integration
- Can be called 1M times in MC simulation
- Outputs parseable by downstream analysis
- Errors communicated clearly to caller

END OF SPECIFICATION

## Appendix A: Calculation Examples
### Example A.1: Simple PV Project (No Debt)
Inputs:

Capacity: 1,000 kWp
CAPEX: €800,000 (all in Month 0)
OPEX: €15,000/year (fixed, 2% escalation)
EEG: €0.073/kWh for 20 years
Production: 1,000 MWh/year (degradation included in volume model)
Lifetime: 25 years
WACC: 6%
No debt (100% equity)
Taxes: 30%
Depreciation: 20 years

Expected Outputs:

NPV: ~€150,000
IRR: ~7.5%
LCOE: ~€0.055/kWh
Payback: ~12 years


## Appendix B: Naming Conventions
### B.1 Variables

_annual: Arrays indexed by year
_monthly: Arrays indexed by month
_total: Aggregated values
_t or _year: Time-indexed values

### B.2 Units

Energy: kWh (kilowatt-hours)
Power: kW (kilowatts)
Currency: € (euros)
Time: months or years (explicit in variable name)
Rates: decimal (0.06 = 6%)

### B.3 Code Style

Follow PEP 8
Max line length: 120 characters
Use snake_case for functions/variables
Use PascalCase for classes


Document Version: 1.0
Date: 2025-01-26
Author: Financial Model Specification Team
Status: Ready for Implementation