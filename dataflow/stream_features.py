import json
import argparse
import statistics as st

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window as beam_window

RAW_SCHEMA = (
    "machine_id:STRING,event_time:TIMESTAMP,temperature:FLOAT,"
    "vibration:FLOAT,rpm:FLOAT,pressure:FLOAT,voltage:FLOAT"
)
# No label column: a streaming job cannot know the future. Phase 4 adds it.
FEAT_SCHEMA = (
    "machine_id:STRING,window_end:TIMESTAMP,temp_mean:FLOAT,temp_max:FLOAT,"
    "vibration_mean:FLOAT,vibration_std:FLOAT,rpm_mean:FLOAT,pressure_mean:FLOAT,"
    "voltage_mean:FLOAT,reading_count:INTEGER"
)


def parse(msg):
    return json.loads(msg.decode("utf-8"))


def summarize(kv, win=beam.DoFn.WindowParam):
    machine_id, readings = kv
    readings = list(readings)
    n = len(readings)

    def col(k):
        return [x[k] for x in readings]

    temp, vib = col("temperature"), col("vibration")
    return {
        "machine_id": machine_id,
        "window_end": win.end.to_utc_datetime().isoformat(),
        "temp_mean": sum(temp) / n,
        "temp_max": max(temp),
        "vibration_mean": sum(vib) / n,
        "vibration_std": st.pstdev(vib) if n > 1 else 0.0,
        "rpm_mean": sum(col("rpm")) / n,
        "pressure_mean": sum(col("pressure")) / n,
        "voltage_mean": sum(col("voltage")) / n,
        "reading_count": n,          # data-quality signal (was the window complete?)
    }


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)   # NOT --project: that name belongs to Beam
    ap.add_argument("--subscription", required=True)
    ap.add_argument("--dataset", default="maintenance")
    args, beam_args = ap.parse_known_args()

    opts = PipelineOptions(
        beam_args,
        streaming=True,
        save_main_session=True,     # ships module-level imports (json, statistics) to the workers
        project=args.project_id,
    )

    with beam.Pipeline(options=opts) as p:
        readings = (
            p
            | "Read" >> beam.io.ReadFromPubSub(subscription=args.subscription)
            | "Parse" >> beam.Map(parse)
        )

        # Branch 1: raw rows straight to BigQuery
        _ = readings | "RawToBQ" >> beam.io.WriteToBigQuery(
            f"{args.project_id}:{args.dataset}.telemetry_raw",
            schema=RAW_SCHEMA,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
        )

        # Branch 2: 2-minute fixed windows per machine → aggregate → BigQuery
        _ = (
            readings
            | "KV" >> beam.Map(lambda r: (r["machine_id"], r))
            | "Window" >> beam.WindowInto(beam_window.FixedWindows(120))  # 120 s
            | "Group" >> beam.GroupByKey()
            | "Summarize" >> beam.Map(summarize)
            | "FeatToBQ" >> beam.io.WriteToBigQuery(
                f"{args.project_id}:{args.dataset}.features_windowed",
                schema=FEAT_SCHEMA,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
            )
        )


if __name__ == "__main__":
    run()