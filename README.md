# Content

This dataset contains 4 years of electrical consumption, generation, pricing, and weather data for Spain. Consumption and generation data was retrieved from ENTSOE, a public portal for Transmission Service Operator (TSO) data. Settlement prices were obtained from the Spanish TSO Red Electric España. Weather data was purchased as part of a personal project from the Open Weather API for the 5 largest cities in Spain and made public here.

# Acknowledgements

This data is publicly available via ENTSOE and REE and may be found in the following links:
- [ENTSOE](https://transparency.entsoe.eu/)
- [REE Market and Prices](https://www.esios.ree.es/en/market-and-prices) 
- [OpenWeatherMap API](https://openweathermap.org/api)

# Colaborators
Nicholas Jhana (Owner)

# Datasets Description

## Dataset - energy 

- **Hourly data, from 01. Jan 2015 to 31 Dec 2018**
- **Target variable: price**
- **Columns explanations:**
  - Time: Datetime index localized to CET 
  - Generation biomass: biomass generation in MW 
  - Generation fossil brown coal/lignite: coal/lignite generation in MW 
  - Generation fossil coal-derived gas: coal gas generation in MW 
  - Generation fossil gas: gas generation in MW 
  - Generation fossil hard coal: coal generation in MW 
  - Generation fossil oil: oil generation in MW 
  - Generation fossil oil shale: shale oil generation in MW 
  - Generation fossil peat: peat generation in MW 
  - Generation geothermal: geothermal generation in MW 
  - Generation hydro pumped storage aggregated: hydro1 generation in MW 
  - Generation hydro pumped storage consumption: hydro2 generation in MW 
  - Generation hydro run-of-river and poundage: hydro3 generation in MW 
  - Generation hydro water reservoir: hydro4 generation in MW 
  - Generation marine: sea generation in MW 
  - Generation nuclear: nuclear generation in MW 
  - Generation other: other generation in MW 
  - Generation other renewable: other renewable generation in MW 
  - Generation solar: solar generation in MW 
  - Generation waste: waste generation in MW 
  - Generation wind offshore: wind offshore generation in MW 
  - Generation wind onshore: wind onshore generation in MW 
  - Forecast solar day ahead: forecasted solar generation 
  - Forecast wind offshore eday ahead: forecasted offshore wind generation 
  - Forecast wind onshore day ahead: forecasted onshore wind generation 
  - Total load forecast: forecasted electrical demand 
  - Total load actual: actual electrical demand 
  - Price day ahead: forecasted price EUR/MWh 
  - Price actual: price in EUR/MWh 

## Dataset - weather 

- **Hourly data, from 01. Jan 2015 to 31 Dec 2018**
- **Target variable: price**
- **Columns explanations:**
  - dt_iso: datetime index localized to CET
  - text_formatcity_name: name of city
  - temp: in k
  - temp_min: minimum in k
  - temp_max: maximum in k
  - pressure: pressure in hPa
  - humidity: humidity in %
  - wind_speed: wind speed in m/s
  - wind_deg: wind direction
  - rain_1h: rain in last hour in mm
  - rain_3h: rain last 3 hours in mm
  - snow_3h: show last 3 hours in mm
  - clouds_all: cloud cover in %
  - vpn_keyweather_id: Code used to describe weather
  - text_formatweather_main: Short description of current weather
  - text_formatweather_description: Long description of current weather
  - text_formatweather_icon: Weather icon code for website
