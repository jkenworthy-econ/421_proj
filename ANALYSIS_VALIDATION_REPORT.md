# Analysis Validation Report
## UC Professor Salary Compression Study (2013-2024)

**Date:** February 3, 2026  
**Status:** ⚠️ CRITICAL FLAW IDENTIFIED

---

## CRITICAL FLAW: Incorrect CPI Data Usage

### The Problem
The analysis uses FRED series **CPALTT01USM657N** (CPI-U less food and energy), but **misinterprets what these values represent**.

**Values in the cpi.csv file:**
- 2021: 0.5685
- 2022: 0.5241  
- 2023: 0.2755
- 2013 (base): 0.1247

**What they actually are:** CPI index levels (or percentage changes in levels), NOT year-over-year inflation rates

**What the analysis assumes:** These are year-over-year inflation rates (e.g., "2024 inflation was 3.73%")

### Why This Matters
The real wage conversion formula used in the analysis is:
```
Real₂₀₁₃ = Nominal × (CPI₂₀₁₃ / CPI_year)
```

This formula assumes the CPI values are comparable price indices. However, if the values represent cumulative changes or different measures, the conversion produces meaningless results.

**Evidence of the error:**
- The analysis claims "2024 inflation spike +3.73%" but the CPI file only goes to 2023 (0.2755 = 27.55%?)
- These percentages don't match known actual US inflation (2023 actual: ~3.4%, not 27.55%)
- The CPI data appears to be FRED's quarterly data or a different series than expected

---

## Findings That ARE Correct

### ✅ VALID: Nominal Wage Analysis (Cells 1-10 of playground.ipynb)

**Post-2023 Salary Decline:**
- First nominal decline in 11-year period: -3.3% (2023→2024)
- This is directly from salary data, no inflation adjustment needed
- **STATUS: CORRECT**

**Within-Rank Salary Compression (Nominal):**
- Associate professors: -23.1% variance reduction (pre-2023 → post-2023)
- Clinical professors: -14.9% variance reduction
- Research professors: -13.6% variance reduction
- Full professors: -3.9% variance reduction
- Assistant professors: +28.8% variance expansion
- **STATUS: CORRECT** (based on coefficient of variation in actual dollars)

**Statistical Significance:**
- Levene's Test shows p < 0.001 for all ranks
- Conclusion: Compression is statistically significant
- **STATUS: CORRECT** (valid hypothesis test on the data)

---

## Findings That ARE QUESTIONABLE/FLAWED

### ❌ INVALID: Real Wages Analysis (Cells 14-20 of playground.ipynb)

**Affected Conclusions:**
1. "Real wages show STRONGER compression than nominal" - **INVALIDATED**
2. "2024 inflation spike (+3.73%)" - **UNSUPPORTED** (no 2024 CPI data)
3. "UC raises insufficient for inflation" - **REQUIRES CORRECT CPI DATA**
4. All comparisons between nominal and real wage compression - **INVALID**

**Root Cause:**
- CPI values are incorrectly interpreted
- Cannot determine correct real wage adjustments
- Cannot validate whether compression persists in "real terms"

---

## What's Missing/Needed for Valid Real Wage Analysis

1. **Correct CPI-U Data** - Obtain actual annual CPI-U values (not FRED code values)
   - Should show ~3-4% for recent years, not 27% or 52%
   
2. **2024 CPI Data** - The file ends at 2023; 2024 data needed for comparing 2023→2024 change

3. **Verify the Data Source** - Check if cpi.csv was transformed or if it's raw FRED output needing conversion

4. **Rerun Analysis** - With correct CPI values, recompute real wages and test for persistence of compression

---

## Summary of Analysis Validity

| Finding | Status | Notes |
|---------|--------|-------|
| Post-2023 nominal salary decline | ✅ CORRECT | Direct salary comparison, no inflation adjustments |
| Within-rank compression (nominal terms) | ✅ CORRECT | Valid dispersion metric (CV) |
| Statistical significance of compression | ✅ CORRECT | Proper hypothesis test with valid p-values |
| Compression "persists in real terms" | ❌ INVALID | Requires correct CPI data |
| Specific real wage values | ❌ INVALID | CPI conversion formula based on wrong data |
| "UC raises insufficient for inflation" | ⚠️ UNVERIFIABLE | Requires correct CPI data to test |

---

## Recommendations

1. **For Current Use:**
   - Cite ONLY the nominal wage findings (compression magnitude, statistical significance)
   - Do NOT cite real wage conclusions until CPI data is corrected
   - The core finding (compression occurs post-2023) stands on nominal wage evidence alone

2. **For Publication/Presentation:**
   - Remove all real wage comparisons from executive summary
   - Note that CPI data is incomplete (ends at 2023, no 2024 value)
   - State: "Real wage analysis pending correct inflation data"

3. **For Future Analysis:**
   - Obtain official CPI-U data from BLS or FRED directly
   - Verify data format before use
   - Include 2024 CPI for complete comparison period

---

## Conclusion

**The core finding about post-2023 wage compression IS VALID** (based on nominal salary data), but the interpretation that this "persists in real terms" **CANNOT BE CONFIRMED** with the current CPI data. 

The analysis would be stronger if limited to nominal wage conclusions with a note that real wage analysis requires verified inflation data.
