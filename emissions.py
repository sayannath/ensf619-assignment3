from codecarbon import OfflineEmissionsTracker
from codecarbon.core import util as codecarbon_util

# Same fix as Assignment 2: under Slurm, CodeCarbon divides the whole machine's CPU TDP by only the
# job's allocated CPUs, which overstates CPU power. This makes it count all of the machine's CPUs.
codecarbon_util.SLURM_JOB_ID = None


def tracked(name, fn, *args, **kwargs):
    """Run fn inside a CodeCarbon tracker and return (result, emissions data)."""
    tracker = OfflineEmissionsTracker(
        project_name=name,
        country_iso_code="CAN",
        region="alberta",
        measure_power_secs=1,
        tracking_mode="process",
        save_to_file=False,
        log_level="error",
    )
    tracker.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        tracker.stop()
    return result, tracker.final_emissions_data
