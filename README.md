# T06 Water Utility Security Analytics

This project is a controlled, synthetic-data prototype for detecting suspicious
vendor remote access and abnormal operational-technology activity in a water
utility setting. It does not connect to a live SCADA or water utility system.

## Quick start

1. Install the packages in `requirements.txt`.
2. Run `python 03_notebooks_or_scripts/run_pipeline.py`.
3. Open `07_dashboard_or_prototype/dashboard.html` in a web browser.

The pipeline recreates the raw data, processed data, models, figures, metrics,
incident timeline, intelligence products, simulation and dashboard.

## Analyst workflow

1. Review the overview and model metrics.
2. Filter suspicious sessions by risk level or vendor.
3. Examine the correlated VENDOR_07/OT_04 incident timeline.
4. Compare the three control scenarios.
5. Decide whether to allow, monitor, investigate or block/isolate the session.

Risk scores of 70 or higher require investigation. Blocking or isolation still
requires human confirmation because this is decision support, not automatic OT control.

## Main limitations

- All data is synthetic and simplified.
- The prototype is not connected to real operational systems.
- Ticket text is generated from a limited set of templates.
- Results show the workflow and decision logic, not production accuracy.
