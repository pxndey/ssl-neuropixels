"""Monopolar-triangulation localization pipeline ported from the sibling `sln`
project (bandpass + CMR + TPCA denoise + monopolar-triangulation). Used to
localize our OWN detected spikes so the result is directly comparable, per spike,
to the SLN model localizations. Modules config/signals/geometry/extraction/tpca/
localization are verbatim copies of /scratch/ap7151/sln/src/preprocessing/*."""
