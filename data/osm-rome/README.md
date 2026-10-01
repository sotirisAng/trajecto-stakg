# OSM Rome

`rome_center.tar.gz` holds the public GPS traces of the Rome city center as downloaded for this work from the
OpenStreetMap API on 15 October 2025: 50 pages of trackpoints (`page_00.gpx` to `page_49.gpx`) for the bounding box
12.44734,41.88217,12.49686,41.90720 (west, south, east, north). `fetch_osm_traces.py` is the download script.

```bash
tar -xzf rome_center.tar.gz      # creates rome_center/, which notebook 0 reads
```

Identifiable traces carry the OSM user name of their contributor, which becomes the person in the KG. Traces without
a user name get a pseudonym from `files/pseudonymous_users.json`.

The traces are © OpenStreetMap contributors and subject to the
[OpenStreetMap terms](https://www.openstreetmap.org/copyright). OSM's API returns traces in pages that change as new
traces are uploaded, so re-running the download gives a different corpus; use the archive to reproduce the thesis.
