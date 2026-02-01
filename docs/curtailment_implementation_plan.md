# Curtailment Implementation Plan

**Version:** 1.0
**Datum:** 2026-01-31
**Status:** Entwurf - Wartet auf Verifizierung

---

## 1. Übersicht

### 1.1 Was ist Curtailment?

Curtailment bezeichnet die zwangsweise Abregelung von Erzeugungsanlagen bei:
- **Netzengpässen** (Einspeisemanagement nach §14 EEG)
- **Negativen Strompreisen** (Marktbasierte Abregelung)
- **Technischen Einschränkungen** (z.B. 70%-Regelung für kleine PV-Anlagen)

### 1.2 Anforderungen

| Anforderung | Beschreibung |
|-------------|--------------|
| **Performance** | < 10ms zusätzliche Berechnungszeit |
| **Monte Carlo Integration** | Alle Curtailment-Parameter sollen variierbar sein |
| **Flexibilität** | Unterstützung verschiedener Curtailment-Modi |
| **Granularität** | Stündlich, monatlich oder jährlich anwendbar |
| **Kompatibilität** | Nahtlose Integration in bestehendes Volume Model Pattern |

### 1.3 Empfohlene Architektur

```
VolumeModel.calculate()
        ↓
[hourly_volumes] ─────────────────────┐
        ↓                             │
CurtailmentModel.apply()              │ Multiplikativ
        ↓                             │
[curtailed_volumes] ──────────────────┘
        ↓
RevenueModel.calculate_hourly()
```

**Begründung:** Curtailment wird als separates Modul implementiert, das zwischen Volume Model und Revenue Model geschaltet wird. Dies ermöglicht:
- Klare Trennung der Verantwortlichkeiten
- Einfaches Testing
- Flexible Konfiguration durch Monte Carlo Framework

---

## 2. Curtailment-Modi

### 2.1 Modus 1: Feste jährliche Raten (Simple)

**Anwendungsfall:** Konservative Schätzung basierend auf historischen Daten

```python
curtailment_params = {
    'mode': 'annual_rates',
    'rates': [0.02, 0.025, 0.03, ...]  # Pro Jahr, 0.02 = 2% Abregelung
}
```

**Berechnung:**
```python
curtailed_production[year] = production[year] * (1 - curtailment_rate[year])
```

### 2.2 Modus 2: Zeitreihen-basiert (CSV)

**Anwendungsfall:** Detaillierte Modellierung mit stündlichen/monatlichen Profilen

```python
curtailment_params = {
    'mode': 'timeseries',
    'csv_path': 'path/to/curtailment_profile.csv',
    'granularity': 'hourly'  # oder 'monthly'
}
```

**CSV-Format:**
```csv
hour,curtailment_factor
0,1.0
1,1.0
...
720,0.85  # 15% Abregelung in dieser Stunde
```

### 2.3 Modus 3: Preisbasiert (Markt)

**Anwendungsfall:** Abregelung bei negativen Strompreisen (§51 EEG)

```python
curtailment_params = {
    'mode': 'price_based',
    'price_threshold': 0.0,  # Abregeln bei Preis < 0
    'curtailment_factor': 1.0  # Vollständige Abschaltung bei Unterschreitung
}
```

**Berechnung:**
```python
for hour in range(n_hours):
    if price[hour] < price_threshold:
        curtailed_production[hour] = production[hour] * (1 - curtailment_factor)
```

### 2.4 Modus 4: Kapazitätsbasiert (70%-Regel)

**Anwendungsfall:** Technische Einspeisebegrenzung

```python
curtailment_params = {
    'mode': 'capacity_limit',
    'limit_factor': 0.70,  # Max 70% der Nennleistung einspeisen
    'applies_to': 'peak_hours'  # oder 'all'
}
```

**Berechnung:**
```python
max_production = capacity * limit_factor
curtailed_production = np.minimum(production, max_production)
```

### 2.5 Modus 5: Stochastisch (Monte Carlo)

**Anwendungsfall:** Unsicherheitsmodellierung für Netzengpässe

```python
curtailment_params = {
    'mode': 'stochastic',
    'base_rate': 0.02,
    'volatility': 0.01,  # Standardabweichung
    'trend': 0.001,  # Jährliche Zunahme der Engpässe
    'correlation_with_production': 0.3  # Höhere Produktion → höheres Curtailment
}
```

---

## 3. Architektur & Dateistruktur

### 3.1 Neue Dateien

```
financial_model/
├── models/
│   └── curtailment_model.py      # NEU: CurtailmentModel Klasse
├── interfaces/
│   └── curtailment_interface.py  # NEU: Abstract Interface (optional)
└── tests/
    └── test_curtailment_model.py # NEU: Unit Tests
```

### 3.2 Zu ändernde Dateien

| Datei | Änderung |
|-------|----------|
| `core/financial_model.py` | Integration nach Volume-Berechnung |
| `schemas/financial_model_input.schema.json` | Neue Parameter-Definition |
| `templates/asset_templates.py` | Standardwerte pro Asset-Typ |
| `tests/test_integration_pv.py` | Integrationstests erweitern |

---

## 4. Detailliertes Design

### 4.1 CurtailmentModel Klasse

```python
# financial_model/models/curtailment_model.py

import numpy as np
from typing import Optional, Union

class CurtailmentModel:
    """
    Berechnet Curtailment (Abregelung) für Erzeugungsanlagen.

    Unterstützt verschiedene Modi:
    - annual_rates: Feste jährliche Abregelungsraten
    - timeseries: Zeitreihen-basierte Profile
    - price_based: Preisabhängige Abregelung
    - capacity_limit: Kapazitätsbegrenzung (z.B. 70%-Regel)
    - stochastic: Stochastische Modellierung für MC
    """

    def __init__(self):
        self._cache = {}

    def apply(
        self,
        volumes: np.ndarray,
        params: dict,
        n_years: int,
        hourly_prices: Optional[np.ndarray] = None,
        capacity_kw: Optional[float] = None
    ) -> np.ndarray:
        """
        Wendet Curtailment auf Produktionsvolumen an.

        Args:
            volumes: Produktionsarray (stündlich oder monatlich)
            params: Curtailment-Parameter aus Input-Dict
            n_years: Projektlaufzeit in Jahren
            hourly_prices: Optional für preisbasiertes Curtailment
            capacity_kw: Optional für kapazitätsbasiertes Curtailment

        Returns:
            np.ndarray: Abgeregelte Volumina (gleiche Shape wie Input)
        """
        mode = params.get('mode', 'annual_rates')

        if mode == 'annual_rates':
            return self._apply_annual_rates(volumes, params, n_years)
        elif mode == 'timeseries':
            return self._apply_timeseries(volumes, params)
        elif mode == 'price_based':
            return self._apply_price_based(volumes, params, hourly_prices)
        elif mode == 'capacity_limit':
            return self._apply_capacity_limit(volumes, params, capacity_kw)
        elif mode == 'stochastic':
            return self._apply_stochastic(volumes, params, n_years)
        else:
            raise ValueError(f"Unknown curtailment mode: {mode}")

    def _apply_annual_rates(
        self,
        volumes: np.ndarray,
        params: dict,
        n_years: int
    ) -> np.ndarray:
        """Wendet jährliche Curtailment-Raten an."""
        rates = params.get('rates', [])

        # Falls einzelner Wert: auf alle Jahre erweitern
        if isinstance(rates, (int, float)):
            rates = [rates] * n_years

        # Erweitere falls nötig (letzten Wert wiederholen)
        while len(rates) < n_years:
            rates.append(rates[-1] if rates else 0.0)

        rates = np.array(rates[:n_years])

        # Bestimme Granularität
        is_hourly = len(volumes) == n_years * 8760

        if is_hourly:
            # Erstelle stündliche Faktoren aus jährlichen Raten
            factors = np.repeat(1 - rates, 8760)
        else:
            # Monatlich: 12 Monate pro Jahr
            factors = np.repeat(1 - rates, 12)

        return volumes * factors

    def _apply_timeseries(
        self,
        volumes: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """Lädt und wendet Zeitreihen-Profile an."""
        csv_path = params.get('csv_path')
        if csv_path is None:
            raise ValueError("csv_path required for timeseries mode")

        # Caching für Performance
        if csv_path not in self._cache:
            self._cache[csv_path] = self._load_curtailment_csv(csv_path)

        factors = self._cache[csv_path]

        # Profile auf Projektlänge erweitern (Wiederholung)
        if len(factors) < len(volumes):
            repeats = int(np.ceil(len(volumes) / len(factors)))
            factors = np.tile(factors, repeats)[:len(volumes)]

        return volumes * factors[:len(volumes)]

    def _apply_price_based(
        self,
        volumes: np.ndarray,
        params: dict,
        hourly_prices: np.ndarray
    ) -> np.ndarray:
        """Preisabhängiges Curtailment (z.B. bei negativen Preisen)."""
        if hourly_prices is None:
            raise ValueError("hourly_prices required for price_based mode")

        threshold = params.get('price_threshold', 0.0)
        curtailment_factor = params.get('curtailment_factor', 1.0)

        # Erstelle Maske für Stunden unter Schwellwert
        below_threshold = hourly_prices < threshold

        # Berechne Faktoren
        factors = np.ones_like(volumes, dtype=float)
        factors[below_threshold[:len(factors)]] = 1 - curtailment_factor

        return volumes * factors

    def _apply_capacity_limit(
        self,
        volumes: np.ndarray,
        params: dict,
        capacity_kw: float
    ) -> np.ndarray:
        """Kapazitätsbegrenzung (z.B. 70%-Regel)."""
        if capacity_kw is None:
            raise ValueError("capacity_kw required for capacity_limit mode")

        limit_factor = params.get('limit_factor', 0.70)
        max_production = capacity_kw * limit_factor

        return np.minimum(volumes, max_production)

    def _apply_stochastic(
        self,
        volumes: np.ndarray,
        params: dict,
        n_years: int
    ) -> np.ndarray:
        """Stochastisches Curtailment für Monte Carlo."""
        base_rate = params.get('base_rate', 0.02)
        volatility = params.get('volatility', 0.01)
        trend = params.get('trend', 0.0)
        seed = params.get('seed')  # Für Reproduzierbarkeit

        if seed is not None:
            np.random.seed(seed)

        # Generiere jährliche Raten mit Trend und Volatilität
        years = np.arange(n_years)
        expected_rates = base_rate + trend * years
        random_component = np.random.normal(0, volatility, n_years)
        rates = np.clip(expected_rates + random_component, 0, 1)

        # Anwenden wie annual_rates
        return self._apply_annual_rates(
            volumes,
            {'rates': rates.tolist()},
            n_years
        )

    def _load_curtailment_csv(self, csv_path: str) -> np.ndarray:
        """Lädt Curtailment-Faktoren aus CSV."""
        import csv

        factors = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Erwarte 'curtailment_factor' Spalte
                factor = float(row.get('curtailment_factor', 1.0))
                factors.append(factor)

        return np.array(factors)

    def get_annual_curtailment_summary(
        self,
        original_volumes: np.ndarray,
        curtailed_volumes: np.ndarray,
        n_years: int
    ) -> dict:
        """
        Berechnet jährliche Curtailment-Statistiken.

        Returns:
            dict mit 'annual_curtailed_mwh', 'annual_curtailment_rates', 'total_curtailment_rate'
        """
        is_hourly = len(original_volumes) == n_years * 8760

        if is_hourly:
            original_annual = np.array([
                np.sum(original_volumes[y*8760:(y+1)*8760])
                for y in range(n_years)
            ])
            curtailed_annual = np.array([
                np.sum(curtailed_volumes[y*8760:(y+1)*8760])
                for y in range(n_years)
            ])
        else:
            original_annual = np.array([
                np.sum(original_volumes[y*12:(y+1)*12])
                for y in range(n_years)
            ])
            curtailed_annual = np.array([
                np.sum(curtailed_volumes[y*12:(y+1)*12])
                for y in range(n_years)
            ])

        curtailed_mwh = (original_annual - curtailed_annual) / 1000  # kWh → MWh

        # Vermeide Division durch Null
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.where(
                original_annual > 0,
                1 - (curtailed_annual / original_annual),
                0
            )

        return {
            'annual_curtailed_mwh': curtailed_mwh,
            'annual_curtailment_rates': rates,
            'total_curtailment_rate': 1 - np.sum(curtailed_volumes) / np.sum(original_volumes)
        }
```

### 4.2 Integration in FinancialModel

**Änderung in `core/financial_model.py`:**

```python
# Nach Zeile ~111 in calculate_hourly():

# Bestehender Code:
if hourly_volumes is None:
    hourly_volumes = self.volume_model.calculate(params["technical"])

# NEU: Curtailment anwenden falls konfiguriert
curtailment_params = params.get("technical", {}).get("curtailment_params")
if curtailment_params is not None:
    from ..models.curtailment_model import CurtailmentModel

    curtailment_model = CurtailmentModel()
    original_volumes = hourly_volumes.copy()  # Für Statistiken

    hourly_volumes = curtailment_model.apply(
        volumes=hourly_volumes,
        params=curtailment_params,
        n_years=n_years,
        hourly_prices=hourly_prices if curtailment_params.get('mode') == 'price_based' else None,
        capacity_kw=params["technical"].get("capacity")
    )

    # Optional: Curtailment-Statistiken speichern
    curtailment_stats = curtailment_model.get_annual_curtailment_summary(
        original_volumes, hourly_volumes, n_years
    )
```

### 4.3 Parameter-Schema Erweiterung

**Änderung in `schemas/financial_model_input.schema.json`:**

```json
{
  "technical": {
    "properties": {
      "curtailment_params": {
        "type": "object",
        "description": "Curtailment configuration for grid constraints",
        "properties": {
          "mode": {
            "type": "string",
            "enum": ["annual_rates", "timeseries", "price_based", "capacity_limit", "stochastic"],
            "description": "Curtailment calculation mode"
          },
          "rates": {
            "oneOf": [
              {"type": "number", "minimum": 0, "maximum": 1},
              {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}}
            ],
            "description": "Curtailment rates for annual_rates mode"
          },
          "csv_path": {
            "type": "string",
            "description": "Path to curtailment profile CSV for timeseries mode"
          },
          "price_threshold": {
            "type": "number",
            "description": "Price threshold for price_based mode (€/kWh)"
          },
          "curtailment_factor": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Curtailment factor when threshold is breached"
          },
          "limit_factor": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Capacity limit factor for capacity_limit mode"
          },
          "base_rate": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Base curtailment rate for stochastic mode"
          },
          "volatility": {
            "type": "number",
            "minimum": 0,
            "description": "Volatility for stochastic mode"
          },
          "trend": {
            "type": "number",
            "description": "Annual trend for stochastic mode"
          },
          "seed": {
            "type": "integer",
            "description": "Random seed for reproducibility in stochastic mode"
          }
        },
        "required": ["mode"]
      }
    }
  }
}
```

---

## 5. Monte Carlo Integration

### 5.1 Variierbare Parameter

Das MC-Framework kann folgende Parameter variieren:

| Parameter | Typ | Beschreibung | Beispiel-Range |
|-----------|-----|--------------|----------------|
| `rates` | Array/Float | Jährliche Abregelungsraten | 0.01 - 0.10 |
| `base_rate` | Float | Basis-Rate für stochastisch | 0.01 - 0.05 |
| `volatility` | Float | Unsicherheit | 0.005 - 0.02 |
| `trend` | Float | Jährliche Zunahme | 0.0 - 0.005 |
| `price_threshold` | Float | Preisschwelle | -0.05 - 0.0 |
| `curtailment_factor` | Float | Abregelungsgrad | 0.5 - 1.0 |
| `seed` | Int | Reproduzierbarkeit | Eindeutige Seeds |

### 5.2 Beispiel: MC-Konfiguration

```python
mc_config = {
    'iterations': 10000,
    'variables': [
        {
            'path': 'technical.curtailment_params.base_rate',
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.05
        },
        {
            'path': 'technical.curtailment_params.trend',
            'distribution': 'triangular',
            'min': 0.0,
            'mode': 0.002,
            'max': 0.005
        }
    ]
}
```

### 5.3 Seed-Management für Reproduzierbarkeit

```python
# Im MC-Framework:
for iteration in range(n_iterations):
    params['technical']['curtailment_params']['seed'] = base_seed + iteration
    results = financial_model.calculate_hourly(params)
```

---

## 6. Performance-Optimierung

### 6.1 Benchmarks (Ziel: < 10ms)

| Operation | Erwartete Zeit | Optimierung |
|-----------|----------------|-------------|
| annual_rates | < 0.1ms | Vektorisiert |
| timeseries | < 0.5ms | CSV-Caching |
| price_based | < 0.2ms | NumPy Masking |
| capacity_limit | < 0.1ms | np.minimum |
| stochastic | < 0.3ms | Vektorisierte RNG |

### 6.2 Optimierungsstrategien

```python
# 1. Pre-Allokation statt Append
factors = np.zeros(n_hours)  # Gut
factors = []  # Schlecht
for ...: factors.append(...)

# 2. Vektorisierte Operationen
factors = np.repeat(1 - rates, 8760)  # Gut
for year in range(n_years):  # Schlecht
    for hour in range(8760):
        factors[year*8760 + hour] = 1 - rates[year]

# 3. Caching für wiederholte Aufrufe
if csv_path not in self._cache:
    self._cache[csv_path] = self._load_csv(csv_path)

# 4. In-Place Operationen wo möglich
volumes *= factors  # Gut (modifiziert Array direkt)
volumes = volumes * factors  # OK (neues Array)
```

### 6.3 Lazy Loading

```python
# CSV nur bei Bedarf laden (nicht bei __init__)
def _apply_timeseries(self, volumes, params):
    if csv_path not in self._cache:
        self._cache[csv_path] = self._load_curtailment_csv(csv_path)
```

---

## 7. Asset-spezifische Defaults

### 7.1 Erweiterung von `templates/asset_templates.py`

```python
ASSET_TEMPLATES = {
    'pv': {
        # ... bestehende Felder ...
        'curtailment_defaults': {
            'mode': 'annual_rates',
            'rates': 0.02,  # 2% typisch für PV in DE
            'typical_range': (0.01, 0.05)
        }
    },
    'wind': {
        # ... bestehende Felder ...
        'curtailment_defaults': {
            'mode': 'annual_rates',
            'rates': 0.03,  # 3% typisch für Wind in DE (Küstennähe höher)
            'typical_range': (0.02, 0.08)
        }
    },
    'heat_network': {
        # ... bestehende Felder ...
        'curtailment_defaults': None  # Kein Curtailment für Wärmenetze
    }
}
```

---

## 8. Test-Strategie

### 8.1 Unit Tests

```python
# tests/test_curtailment_model.py

import numpy as np
import pytest
from financial_model.models.curtailment_model import CurtailmentModel

class TestCurtailmentModel:

    @pytest.fixture
    def model(self):
        return CurtailmentModel()

    @pytest.fixture
    def hourly_volumes(self):
        """25 Jahre stündliche Produktion."""
        return np.ones(25 * 8760) * 100  # 100 kWh/h

    def test_annual_rates_single_value(self, model, hourly_volumes):
        """Test mit einzelnem Curtailment-Wert."""
        params = {'mode': 'annual_rates', 'rates': 0.05}
        result = model.apply(hourly_volumes, params, n_years=25)

        assert len(result) == len(hourly_volumes)
        np.testing.assert_allclose(result, hourly_volumes * 0.95)

    def test_annual_rates_array(self, model, hourly_volumes):
        """Test mit jährlich variierenden Raten."""
        rates = [0.01 * i for i in range(25)]  # 0%, 1%, 2%, ..., 24%
        params = {'mode': 'annual_rates', 'rates': rates}
        result = model.apply(hourly_volumes, params, n_years=25)

        for year in range(25):
            year_slice = slice(year * 8760, (year + 1) * 8760)
            expected_factor = 1 - 0.01 * year
            np.testing.assert_allclose(
                result[year_slice],
                hourly_volumes[year_slice] * expected_factor
            )

    def test_capacity_limit(self, model, hourly_volumes):
        """Test 70%-Regel."""
        params = {'mode': 'capacity_limit', 'limit_factor': 0.70}
        # Produktion variiert: 0-200 kWh/h
        varying_volumes = np.sin(np.linspace(0, 10*np.pi, len(hourly_volumes))) * 100 + 100

        result = model.apply(varying_volumes, params, n_years=25, capacity_kw=100)

        assert np.all(result <= 70)  # Max 70 kWh/h
        assert np.any(result < varying_volumes)  # Einige wurden begrenzt

    def test_price_based_curtailment(self, model, hourly_volumes):
        """Test preisbasierte Abregelung bei negativen Preisen."""
        params = {
            'mode': 'price_based',
            'price_threshold': 0.0,
            'curtailment_factor': 1.0
        }

        # 10% negative Preise
        prices = np.random.randn(len(hourly_volumes))
        negative_count = np.sum(prices < 0)

        result = model.apply(hourly_volumes, params, n_years=25, hourly_prices=prices)

        # Bei negativen Preisen: Produktion = 0
        assert np.sum(result == 0) == negative_count

    def test_stochastic_reproducibility(self, model, hourly_volumes):
        """Test Reproduzierbarkeit mit Seed."""
        params = {
            'mode': 'stochastic',
            'base_rate': 0.03,
            'volatility': 0.01,
            'seed': 42
        }

        result1 = model.apply(hourly_volumes, params, n_years=25)
        result2 = model.apply(hourly_volumes, params, n_years=25)

        np.testing.assert_array_equal(result1, result2)

    def test_performance(self, model, hourly_volumes):
        """Performance-Test: < 10ms."""
        import time

        params = {'mode': 'annual_rates', 'rates': 0.03}

        start = time.perf_counter()
        for _ in range(100):
            model.apply(hourly_volumes, params, n_years=25)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 10, f"Zu langsam: {avg_ms:.2f}ms"

    def test_curtailment_summary(self, model, hourly_volumes):
        """Test Statistik-Berechnung."""
        params = {'mode': 'annual_rates', 'rates': 0.05}
        result = model.apply(hourly_volumes, params, n_years=25)

        stats = model.get_annual_curtailment_summary(
            hourly_volumes, result, n_years=25
        )

        assert 'annual_curtailment_rates' in stats
        np.testing.assert_allclose(stats['annual_curtailment_rates'], 0.05, atol=0.001)
        np.testing.assert_allclose(stats['total_curtailment_rate'], 0.05, atol=0.001)
```

### 8.2 Integrationstests

```python
# Erweiterung von tests/test_integration_pv.py

def test_pv_with_curtailment():
    """PV-Projekt mit 3% Curtailment."""
    params = get_pv_test_params()
    params['technical']['curtailment_params'] = {
        'mode': 'annual_rates',
        'rates': 0.03
    }

    model = FinancialModel(asset_type='pv')
    results = model.calculate_hourly(params, hourly_volumes)

    # NPV sollte ~3% niedriger sein als ohne Curtailment
    results_no_curtailment = model.calculate_hourly(
        get_pv_test_params(), hourly_volumes
    )

    reduction_ratio = results['kpis']['npv_project'] / results_no_curtailment['kpis']['npv_project']
    assert 0.95 < reduction_ratio < 0.99  # Erwarte ~3% Reduktion
```

---

## 9. Output-Erweiterung

### 9.1 Erweiterte Ergebnisstruktur

```python
results = {
    'kpis': {...},  # Bestehende KPIs

    'curtailment': {  # NEU
        'enabled': True,
        'mode': 'annual_rates',
        'annual_curtailed_mwh': np.ndarray,  # Pro Jahr
        'annual_curtailment_rates': np.ndarray,  # Pro Jahr
        'total_curtailment_rate': float,  # Gesamt
        'total_curtailed_mwh': float,  # Gesamt MWh
        'revenue_loss_eur': float  # Geschätzter Umsatzverlust
    },

    'cash_flows': {...}  # Bestehende Cashflows
}
```

---

## 10. Dokumentation

### 10.1 Docstrings (Google Style)

```python
def apply(
    self,
    volumes: np.ndarray,
    params: dict,
    n_years: int,
    hourly_prices: Optional[np.ndarray] = None,
    capacity_kw: Optional[float] = None
) -> np.ndarray:
    """
    Wendet Curtailment auf Produktionsvolumen an.

    Curtailment reduziert die Einspeisemenge basierend auf verschiedenen
    Faktoren wie Netzengpässen, negativen Strompreisen oder technischen
    Begrenzungen.

    Args:
        volumes: Produktionsarray in kWh. Shape: (n_hours,) für stündlich
            oder (n_months,) für monatlich.
        params: Curtailment-Konfiguration mit mindestens 'mode' Schlüssel.
            Siehe Klassenbeispiele für Modi-spezifische Parameter.
        n_years: Projektlaufzeit in Jahren. Wird für Zeitreihen-Erweiterung
            und stochastische Berechnung benötigt.
        hourly_prices: Stündliche Strompreise in €/kWh. Nur erforderlich
            für mode='price_based'.
        capacity_kw: Nennleistung der Anlage in kW. Nur erforderlich
            für mode='capacity_limit'.

    Returns:
        np.ndarray: Abgeregelte Volumina mit gleicher Shape wie Eingabe.
            Werte sind immer <= Eingabewerte.

    Raises:
        ValueError: Wenn erforderliche Parameter für den gewählten Modus
            fehlen oder unbekannter Modus angegeben wird.

    Example:
        >>> model = CurtailmentModel()
        >>> volumes = np.ones(25 * 8760) * 100  # 100 kWh/h
        >>> params = {'mode': 'annual_rates', 'rates': 0.03}
        >>> curtailed = model.apply(volumes, params, n_years=25)
        >>> assert np.all(curtailed == 97)  # 3% weniger
    """
```

---

## 11. Implementierungs-Checkliste

### Phase 1: Core Implementation
- [ ] `curtailment_model.py` erstellen
- [ ] Alle 5 Modi implementieren
- [ ] Unit Tests für jeden Modus

### Phase 2: Integration
- [ ] `financial_model.py` erweitern
- [ ] Schema aktualisieren
- [ ] Asset Templates erweitern

### Phase 3: Testing & Optimierung
- [ ] Integrationstests
- [ ] Performance-Benchmarks
- [ ] Edge-Case-Handling

### Phase 4: Dokumentation
- [ ] Docstrings vervollständigen
- [ ] README-Beispiele
- [ ] CLAUDE.md aktualisieren

---

## 12. Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Performance > 10ms bei CSV | Niedrig | Hoch | Caching, Lazy Loading |
| Inkonsistente Granularität | Mittel | Mittel | Automatische Erkennung |
| MC-Integration komplex | Niedrig | Mittel | Klare Parameter-Struktur |
| Edge Cases (0 Produktion) | Mittel | Niedrig | Defensive Programmierung |

---

## 13. Offene Fragen

1. **Entschädigungszahlungen:** Sollen Kompensationszahlungen für Curtailment (nach EEG §15) als separate Einnahme modelliert werden? - Ja

2. **Regionale Profile:** Sollen vordefinierte Curtailment-Profile pro Region (Norddeutschland = höher) bereitgestellt werden? - Nein

3. **Korrelation mit Produktion:** Im Modus `stochastic` - soll Curtailment mit hoher Produktion korreliert werden (realistischer)? - Ja, mit einem Eingabeparameter, der das steigende Verhältnis definiert

4. **Negative Preise - Dauer:** Bei preisbasiertem Curtailment - soll die 6-Stunden-Regel nach §51 EEG berücksichtigt werden? - DEr Zeitraum ist aktuell auf 3 Stunden gesenkt worden. Ja, implementiere dies ebenfalls, udn stelle mit unit tests die ordungsgemäße Implementation sicher.

