import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import omero_toolbox as omero_tb
    import analysis_functions
    from time import sleep

    return analysis_functions, mo, np, omero_tb, sleep


@app.cell
def _(mo):
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
                },
            ),
            analysis_parameters=mo.ui.dictionary(
                label="Analysis parameters",
                elements={
                    "min_sigma": mo.ui.number(
                        value=10.0,
                        label="Min sigma",
                        start=1.0,
                        stop=100.0,
                    ),
                    "max_sigma": mo.ui.number(
                        value=80.0,
                        label="Max sigma",
                        start=2.0,
                        stop=200.0,
                    ),
                    "channel": mo.ui.number(
                        value=0,
                        label="Channel to analyze",
                        start=0,
                        stop=100,
                        step=1,
                    ),
                    "border_to_exclude": mo.ui.number(
                        value=0,
                        label="Border exclusion size",
                        start=0,
                        stop=200,
                        step=1,
                    ),
                },
            ),
        )
        .form()
    )

    analysis_form


@app.cell
def _(
    mo,
    analysis_functions,
    np,
    omero_tb,
    analysis_form,
):
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

            try:
                dataset = omero_tb.get_dataset(
                    connection=conn, dataset_id=data_params["dataset_id"]
                )
                dataset_images = omero_tb.get_dataset_images(dataset=dataset)

                for image in dataset_images:
                    image_name = image.getName()
                    if not image_name.endswith("SIR.dv"):
                        continue
                    _spinner.update(f"Analyzing: {image_name}")
                    image_data = omero_tb.get_intensities(
                        image, c_range=analysis_params["channel"]
                    )
                    image_data = analysis_functions.rescale_SIM(image_data)
                    image_data = np.squeeze(image_data)
                    spots = analysis_functions.find_spots(
                        image_data,
                        min_sigma=analysis_params["min_sigma"],
                        max_sigma=analysis_params["max_sigma"],
                    )
                    points_with_zc = []
                    points_preview = []
                    for spot in spots:
                        points_with_zc.append(
                            omero_tb.create_shape_point(
                                x_pos=spot[2],
                                y_pos=spot[1],
                                z_pos=spot[0],
                                c_pos=analysis_params["channel"],
                            )
                        )
                        points_preview.append(
                            omero_tb.create_shape_point(x_pos=spot[2], y_pos=spot[1])
                        )

                    omero_tb.create_roi(conn, image, points_with_zc)
                    _spinner.update(
                        f"Found {len(points_with_zc)} spots. Creating preview image"
                    )
                    image_data_mip = np.max(image_data, 0)
                    image_mip = omero_tb.create_image_from_numpy_array(
                        connection=conn,
                        data=image_data_mip,
                        image_name=f"{image_name[:-3]}_preview_mip",
                        dataset=dataset,
                        source_image_id=image.getId(),
                        channels_list=[analysis_params["channel"]],
                    )
                    image_mip = omero_tb.get_image(conn, image_mip.getId())
                    omero_tb.create_roi(conn, image_mip, points_preview)

                _spinner.update("✅ **Analysis Complete!**")

            except Exception as e:
                _spinner.update(f"❌ **Error**: {str(e)}")

            finally:
                conn.close()


if __name__ == "__main__":
    app.run()
