import marimo

app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from omero.model import PointI

    import analysis_functions
    import omero_toolbox as omero_tb

    # Default parameters
    default_analysis_parameters = {
        "properties": (  # Properties to measure
            "label",
            "area",
            "filled_area",
            "major_axis_length",
            "centroid",
            "weighted_centroid",
            "equivalent_diameter",
            "max_intensity",
            "mean_intensity",
            "min_intensity",
            # "coords",
        ),
        "roi_size_xy": 56,
        "roi_size_z": 20,
        "domain_min_volume_px": 210,  # Minimum volume for the regions
        "subdomain_min_volume_px": 38,  # Minimum volume for the regions
        "sigma": 0.5,
        "pixel_size": (0.125, 0.039, 0.039),  # as ZYX
    }
    return (
        mo,
        np,
        pd,
        omero_tb,
        analysis_functions,
        PointI,
        default_analysis_parameters,
    )


@app.cell
def _(default_analysis_parameters, mo):
    analysis_form = (
        mo.md(
            """Fill the analysis parameters:

            {connection_parameters}
            {data_parameters}
            {analysis_parameters}
            """
        )
        .batch(
            connection_parameters=mo.ui.dictionary(
                label="Connection parameters",
                elements={
                    "omero_username": mo.ui.text(label="OMERO username"),
                    "omero_password": mo.ui.text(
                        label="OMERO password",
                        kind="password",
                    ),
                    "omero_host": mo.ui.text(
                        value="omero.mri.cnrs.fr",
                        label="OMERO server URL",
                    ),
                    "omero_port": mo.ui.number(
                        value=4064,
                        start=1,
                        step=1,
                        label="OMERO port",
                    ),
                    "omero_group": mo.ui.text(value="CND project", label="Group"),
                    "connection_secured": mo.ui.checkbox(
                        value=True,
                        label="Secured connection",
                    ),
                },
            ),
            data_parameters=mo.ui.dictionary(
                label="Data parameters",
                elements={
                    "dataset_id": mo.ui.number(start=1, step=1, label="Dataset ID"),
                    "channel": mo.ui.number(
                        value=0,
                        label="Channel to analyze",
                        start=0,
                        stop=100,
                        step=1,
                    ),
                    "image_name_filter": mo.ui.text(
                        value="SIR.dv",
                        label="Image name filter",
                    ),
                },
            ),
            analysis_parameters=mo.ui.dictionary(
                label="Analysis parameters",
                elements={
                    "sigma": mo.ui.number(
                        value=default_analysis_parameters["sigma"],
                        label="Sigma",
                        start=0.0,
                        stop=10.0,
                    ),
                    "properties": mo.ui.multiselect(
                        options=default_analysis_parameters["properties"],
                        value=default_analysis_parameters["properties"],
                        label="Properties",
                    ),
                    "roi_size_xy": mo.ui.number(
                        value=default_analysis_parameters["roi_size_xy"],
                        label="ROI size (px)",
                        start=10,
                        stop=1000,
                    ),
                    "roi_size_z": mo.ui.number(
                        value=default_analysis_parameters["roi_size_z"],
                        label="ROI size (px)",
                        start=6,
                        stop=500,
                    ),
                    "domain_min_volume_px": mo.ui.number(
                        value=default_analysis_parameters["domain_min_volume_px"],
                        label="Domain min volume px",
                        start=11,
                        stop=1001,
                    ),
                    "subdomain_min_volume_px": mo.ui.number(
                        value=default_analysis_parameters["subdomain_min_volume_px"],
                        label="Subdomain min volume px",
                        start=1,
                        stop=1001,
                    ),
                },
            ),
        )
        .form()
    )

    analysis_form


@app.cell
def _(PointI, analysis_form, analysis_functions, mo, np, pd, omero_tb):
    if analysis_form.value is None:
        mo.md("### Waiting for input.")

    else:
        conn_params = analysis_form.value["connection_parameters"]
        data_params = analysis_form.value["data_parameters"]
        analysis_params = analysis_form.value["analysis_parameters"]

        with mo.status.spinner(title="Connecting to OMERO...") as _spinner:
            conn = omero_tb.open_connection(
                username=conn_params["omero_username"],
                password=conn_params["omero_password"],
                host=conn_params["omero_host"],
                port=conn_params["omero_port"],
                group=conn_params["omero_group"],
                secure=conn_params["connection_secured"],
            )

            if conn.connect():
                _spinner.update("Connected to OMERO.")
            else:
                _spinner.update("Failed to connect to OMERO.")
                raise Exception("Connection failed")

            roi_service = conn.getRoiService()

            try:
                dataset = omero_tb.get_dataset(
                    connection=conn, dataset_id=data_params["dataset_id"]
                )
                dataset_images = omero_tb.get_dataset_images(dataset=dataset)

                dataset_df = pd.DataFrame()

                for image in dataset_images:
                    image_name = image.getName()
                    if data_params["image_name_filter"] not in image_name:
                        continue
                    _spinner.update(f"Analyzing: {image_name}")

                    image_df = pd.DataFrame()

                    result = roi_service.findByImage(image.getId(), None)
                    for source_roi in result.rois:
                        for source_shape in source_roi.iterateShapes():
                            if not isinstance(source_shape, PointI):
                                continue
                            try:
                                shape_comment = source_shape.getTextValue()._val
                            except AttributeError:
                                shape_comment = None

                            # Get image dimensions
                            size_x = image.getSizeX()
                            size_y = image.getSizeY()
                            size_z = image.getSizeZ()

                            # Calculate ranges
                            x_range = (
                                int(
                                    source_shape.getX().getValue()
                                    - analysis_params["roi_size_xy"] / 2
                                ),
                                int(
                                    source_shape.getX().getValue()
                                    + analysis_params["roi_size_xy"] / 2
                                ),
                            )
                            y_range = (
                                int(
                                    source_shape.getY().getValue()
                                    - analysis_params["roi_size_xy"] / 2
                                ),
                                int(
                                    source_shape.getY().getValue()
                                    + analysis_params["roi_size_xy"] / 2
                                ),
                            )
                            z_range = (
                                int(
                                    source_shape.getTheZ().getValue()
                                    - analysis_params["roi_size_z"] / 2
                                ),
                                int(
                                    source_shape.getTheZ().getValue()
                                    + analysis_params["roi_size_z"] / 2
                                ),
                            )

                            # Skip if ranges are outside image bounds
                            if (
                                x_range[0] < 0
                                or x_range[1] > size_x
                                or y_range[0] < 0
                                or y_range[1] > size_y
                                or z_range[0] < 0
                                or z_range[1] > size_z
                            ):
                                print(f"ROI {shape_comment} out of bounds, skipping")
                                continue

                            roi_intensities = omero_tb.get_intensities(
                                image=image,
                                c_range=data_params["channel"],
                                x_range=x_range,
                                y_range=y_range,
                                z_range=z_range,
                            )

                            roi_intensities = analysis_functions.rescale_SIM(
                                roi_intensities, out_range="uint16"
                            )

                            # Transpose from zctyx to czyx using t=0
                            roi_intensities = np.transpose(
                                roi_intensities[:, :, 0, :, :], (1, 0, 2, 3)
                            )

                            voxel_size = omero_tb.get_pixel_size(image, order="ZYX")

                            shape_df, domain_labels, subdomain_labels = (
                                analysis_functions.process_image(
                                    image=roi_intensities,
                                    domain_properties=analysis_params["properties"],
                                    subdomain_properties=analysis_params[
                                        "properties"
                                    ],
                                    voxel_size=voxel_size,
                                    sigma=analysis_params["sigma"],
                                    min_volume=analysis_params[
                                        "domain_min_volume_px"
                                    ],
                                    subdomain_min_volume=analysis_params[
                                        "subdomain_min_volume_px"
                                    ],
                                    binarize=False,
                                )
                            )

                            # Correct the centroids position in the dataframe:
                            shape_df["centroid-0"] = (
                                shape_df["centroid-0"] + z_range[0]
                            )
                            shape_df["weighted_centroid-0"] = (
                                shape_df["weighted_centroid-0"] + z_range[0]
                            )
                            shape_df["centroid-1"] = (
                                shape_df["centroid-1"] + y_range[0]
                            )
                            shape_df["weighted_centroid-1"] = (
                                shape_df["weighted_centroid-1"] + y_range[0]
                            )
                            shape_df["centroid-2"] = (
                                shape_df["centroid-2"] + x_range[0]
                            )
                            shape_df["weighted_centroid-2"] = (
                                shape_df["weighted_centroid-2"] + x_range[0]
                            )

                            domain_masks = (
                                omero_tb.create_shapes_mask_from_labels_image_3d(
                                    labels_3d=domain_labels[
                                        0
                                    ],  # TODO: implement channel handling
                                    c_pos=data_params["channel"],
                                    x_pos=x_range[0],
                                    y_pos=y_range[0],
                                    z_pos=z_range[0],
                                    fill_color=(0, 255, 0, 100),
                                )
                            )

                            try:
                                domain_roi = omero_tb.create_roi(
                                    connection=conn,
                                    image=image,
                                    shapes=domain_masks[1],
                                    name=f"{shape_comment}_domain",
                                    description=f"source roi_id: {source_roi.getId()}",
                                )
                            except KeyError:
                                continue

                            subdomain_masks = (
                                omero_tb.create_shapes_mask_from_labels_image_3d(
                                    labels_3d=subdomain_labels[0],
                                    c_pos=data_params["channel"],
                                    x_pos=x_range[0],
                                    y_pos=y_range[0],
                                    z_pos=z_range[0],
                                    fill_color=(255, 0, 0, 100),
                                )
                            )
                            for label, masks in subdomain_masks.items():
                                omero_tb.create_roi(
                                    connection=conn,
                                    image=image,
                                    shapes=masks,
                                    name=f"{shape_comment}_subdomain:{label}",
                                    description=f"source roi_id: {source_roi.getId()}",
                                )

                            shape_df.insert(
                                loc=0,
                                column="source_shape_id",
                                value=source_shape.getId().getValue(),
                            )
                            shape_df.insert(
                                loc=0,
                                column="source_roi_id",
                                value=source_roi.getId().getValue(),
                            )
                            shape_df.insert(
                                loc=0,
                                column="roi_id",
                                value=domain_roi.getId().getValue(),
                            )

                            image_df = pd.concat(
                                [image_df, shape_df], ignore_index=True
                            )

                    image_df.insert(loc=0, column="image_id", value=image.getId())

                    dataset_df = pd.concat([dataset_df, image_df], ignore_index=True)

                dataset_df.insert(
                    loc=0,
                    column="dataset_id",
                    value=dataset.getId(),
                )

                tads_omero_table = omero_tb.create_annotation_table_from_df(
                    connection=conn,
                    dataframe=dataset_df,
                    table_name=f"dataset_{dataset.getId()}_tads",
                    namespace="tads",
                    table_description=f"Table of TADs for datasetid: {dataset.getId()}",
                )

                omero_tb.link_annotation(dataset, tads_omero_table)

            except Exception as e:
                print(f"Error processing image {image.getId()}: {e}")
                raise e

            finally:
                conn.close()


if __name__ == "__main__":
    app.run()
