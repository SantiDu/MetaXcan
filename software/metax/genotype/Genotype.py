import logging

import numpy

from ..misc import GWASAndModels

class GF(object):
    RSID=0
    CHROMOSOME=1
    POSITION=2
    REF_ALLELE=3
    ALT_ALLELE=4
    FREQUENCY=5
    FIRST_DOSAGE=6

def force_mapped_metadata(generator, sep):
    for line in generator:
        varid = line[GF.RSID]
        allele_0, allele_1 = line[GF.REF_ALLELE], line[GF.ALT_ALLELE]
        comps = varid.split(sep)
        f = line[GF.FREQUENCY]
        d = line[GF.FIRST_DOSAGE:]

        allele_swap, strand_swap = GWASAndModels.match_alleles(allele_0, allele_1, comps[2], comps[3])
        if not allele_swap or not strand_swap:
            continue

        pos = int(comps[1])
        chr = comps[0]
        if strand_swap == -1:
            allele_0 = comps[2]
            allele_1 = comps[3]
        if allele_swap == -1:
            allele_0, allele_1 = allele_1, allele_0
            f = 1-f
            d = tuple(map(lambda x:2-x, d))
        yield (varid, chr, pos, allele_0, allele_1, f) + d


def impute_dosage(d, mode="none", threshold=0.05, varid="unknown"):
    """Impute missing (NaN) dosage values for a SNP.

    Parameters
    ----------
    d : numpy.ndarray
        Dosage array (one value per sample). May contain NaN for samples
        with missing genotype probabilities.
    mode : str
        Imputation mode: "none" (no-op) or "mean" (replace NaN with SNP mean).
    threshold : float
        Maximum fraction of samples with missing dosage allowed before
        imputation is skipped. If frac_missing >= threshold, a warning is
        logged and NaN values are kept (the SNP may have a systemic problem).
    varid : str
        Variant identifier, used in warning messages for debugging.

    Returns
    -------
    numpy.ndarray
        Dosage array with NaN values imputed (or kept if threshold exceeded).
    """
    if mode == "none":
        return d

    n_nan = int(numpy.isnan(d).sum())
    if n_nan == 0:
        return d

    n_total = len(d)
    frac_missing = n_nan / n_total

    if frac_missing >= threshold:
        logging.warning(
            "SNP %s: %.2f%% samples missing dosage (threshold: %.2f%%), "
            "keeping NaN — SNP may have a systemic problem",
            varid, frac_missing * 100, threshold * 100
        )
        return d

    # frac_missing < threshold: impute NaN to the mean of non-NaN values
    snp_mean = numpy.nanmean(d)
    if numpy.isnan(snp_mean):
        # All values are NaN (shouldn't happen if frac_missing < threshold
 # and n_nan > 0, but guard anyway)
        logging.warning(
            "SNP %s: all samples missing dosage, keeping NaN", varid
        )
        return d

    d = numpy.nan_to_num(d, nan=snp_mean)
    logging.log(8,
        "SNP %s: imputed %d/%d (%.4f%%) missing dosages to mean=%.6f",
        varid, n_nan, n_total, frac_missing * 100, snp_mean
    )
    return d