"""PVGIS API client for fetching solar radiation and PV production data."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


class PVGISClientError(Exception):
    """Base exception for PVGIS client errors."""

    pass


class PVGISRateLimitError(PVGISClientError):
    """Raised when rate limit is exceeded."""

    pass


class PVGISLocationError(PVGISClientError):
    """Raised when location is outside PVGIS coverage."""

    pass


class PVGISClient:
    """
    HTTP client for the PVGIS API.

    Handles requests to the EU Joint Research Centre's Photovoltaic
    Geographical Information System API with rate limiting and caching.

    Args:
        base_url: PVGIS API base URL.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries on rate limit.
        cache_dir: Directory for disk cache. None disables caching.
        cache_ttl_days: Cache time-to-live in days.

    Example:
        >>> client = PVGISClient()
        >>> data = client.get_tmy_hourly(lat=52.52, lon=13.405)
        >>> len(data['hourly'])
        8760  # hours per year
    """

    BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_3"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        cache_ttl_days: int = 30,
    ) -> None:
        """Initialize the PVGIS client."""
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl_days = cache_ttl_days

        # Setup cache directory - default to financial_model/data/pvgis_cache
        if cache_dir is None:
            # Use data directory relative to this module
            module_dir = Path(__file__).parent.parent
            self.cache_dir = module_dir / "data" / "pvgis_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_tmy_hourly(
        self,
        lat: float,
        lon: float,
        peakpower: float = 1.0,
        loss: float = 14.0,
        angle: float | None = None,
        aspect: float = 0.0,
        startyear: int | None = None,
        endyear: int | None = None,
    ) -> dict[str, Any]:
        """
        Get hourly TMY (Typical Meteorological Year) data with PV calculation.

        Uses the seriescalc endpoint to get hourly production data based on
        TMY weather data. Returns 8760 hourly values representing a typical year.

        Args:
            lat: Latitude in decimal degrees (south is negative).
            lon: Longitude in decimal degrees (west is negative).
            peakpower: Nominal PV system power in kW.
            loss: System losses in percent (default: 14%).
            angle: Panel tilt angle in degrees. None for optimal.
            aspect: Panel azimuth (0=south, 90=west, -90=east).
            startyear: Start year for TMY calculation (optional).
            endyear: End year for TMY calculation (optional).

        Returns:
            Dictionary containing:
                - hourly: List of hourly records with P (power in W), G(i), T2m, etc.
                - inputs: Input parameters used
                - meta: Metadata about the calculation

        Raises:
            PVGISLocationError: If location is outside PVGIS coverage.
            PVGISRateLimitError: If rate limit exceeded after retries.
            PVGISClientError: For other API errors.
        """
        params: dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "peakpower": peakpower,
            "loss": loss,
            "aspect": aspect,
            "outputformat": "json",
            "pvcalculation": 1,
            "usehorizon": 1,
        }

        if angle is not None:
            params["angle"] = angle
        else:
            params["optimalinclination"] = 1

        if startyear is not None:
            params["startyear"] = startyear
        if endyear is not None:
            params["endyear"] = endyear

        # Check cache first
        cache_key = self._get_cache_key("seriescalc", params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        # Make API request
        response = self._make_request("seriescalc", params)

        # Parse and restructure response
        result = self._parse_tmy_response(response)

        # Cache the result
        self._write_cache(cache_key, result)

        return result

    def get_pv_monthly(
        self,
        lat: float,
        lon: float,
        peakpower: float = 1.0,
        loss: float = 14.0,
        angle: float | None = None,
        aspect: float = 0.0,
    ) -> dict[str, Any]:
        """
        Get monthly PV production estimates.

        Uses the PVcalc endpoint to get monthly and annual totals.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            peakpower: Nominal PV system power in kW.
            loss: System losses in percent.
            angle: Panel tilt angle. None for optimal.
            aspect: Panel azimuth (0=south).

        Returns:
            Dictionary with monthly production data.
        """
        params: dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "peakpower": peakpower,
            "loss": loss,
            "aspect": aspect,
            "outputformat": "json",
        }

        if angle is not None:
            params["angle"] = angle
        else:
            params["optimalinclination"] = 1
            params["optimalangles"] = 1

        cache_key = self._get_cache_key("PVcalc", params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        response = self._make_request("PVcalc", params)

        self._write_cache(cache_key, response)

        return response

    def _make_request(
        self, endpoint: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Make HTTP request to PVGIS API with retry logic.

        Args:
            endpoint: API endpoint name.
            params: Query parameters.

        Returns:
            JSON response as dictionary.

        Raises:
            PVGISClientError: On API errors.
        """
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(
                    url, params=params, timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 429:
                    # Rate limit - exponential backoff
                    if attempt < self.max_retries:
                        wait_time = 0.1 * (2**attempt)
                        time.sleep(wait_time)
                        continue
                    raise PVGISRateLimitError(
                        f"Rate limit exceeded after {self.max_retries} retries"
                    )

                if response.status_code == 529:
                    # Server overloaded
                    if attempt < self.max_retries:
                        time.sleep(1.0)
                        continue
                    raise PVGISClientError("Server overloaded, try again later")

                if response.status_code == 400:
                    # Bad request - likely invalid location
                    error_msg = self._extract_error_message(response)
                    if "location" in error_msg.lower() or "outside" in error_msg.lower():
                        raise PVGISLocationError(
                            f"Location ({params.get('lat')}, {params.get('lon')}) "
                            f"is outside PVGIS coverage: {error_msg}"
                        )
                    raise PVGISClientError(f"Bad request: {error_msg}")

                raise PVGISClientError(
                    f"API error {response.status_code}: {response.text[:200]}"
                )

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    continue
                raise PVGISClientError(
                    f"Request timeout after {self.timeout}s"
                )
            except requests.exceptions.RequestException as e:
                raise PVGISClientError(f"Network error: {e}")

        raise PVGISClientError("Max retries exceeded")

    def _parse_tmy_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Parse and restructure TMY/seriescalc response.

        Args:
            response: Raw API response.

        Returns:
            Structured response with hourly data.
        """
        outputs = response.get("outputs", {})
        hourly_data = outputs.get("hourly", [])

        return {
            "hourly": hourly_data,
            "inputs": response.get("inputs", {}),
            "meta": response.get("meta", {}),
        }

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        """Extract error message from response."""
        try:
            data = response.json()
            return data.get("message", data.get("error", str(data)))
        except (ValueError, KeyError):
            return response.text[:200]

    def _get_cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        """
        Generate cache key from endpoint and parameters.

        Args:
            endpoint: API endpoint.
            params: Request parameters.

        Returns:
            SHA256 hash string for cache filename.
        """
        # Sort params for consistent hashing
        sorted_params = sorted(params.items())
        key_string = f"{endpoint}:{sorted_params}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        """
        Read cached response if valid.

        Args:
            cache_key: Cache key hash.

        Returns:
            Cached data or None if not found/expired.
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check TTL
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days > self.cache_ttl_days:
            cache_file.unlink()
            return None

        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _write_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        """
        Write response to cache.

        Args:
            cache_key: Cache key hash.
            data: Data to cache.
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except IOError:
            # Cache write failure is not critical
            pass

    def clear_cache(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache files removed.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
