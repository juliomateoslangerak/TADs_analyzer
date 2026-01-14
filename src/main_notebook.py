import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ### Import the necessary libraries
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import omero_toolbox as omero_tb
    import analysis_functions
    from skimage.feature import blob_log
    return analysis_functions, mo, np, omero_tb


@app.cell(hide_code=True)
def _(mo):
    analysis_params = mo.ui.array([
        mo.ui.number(value=10.0, label="Min sigma", start=1.0, stop=100.0),
        mo.ui.number(value=80.0, label="Max sigma", start=2.0, stop=200.0),
        mo.ui.number(value=0, label="channel to analyze", start=0, stop=10, step=1),
        mo.ui.number(value=0, label="border to exclued", start=0, stop=200, step=1),
    ])
    analysis_params
    return (analysis_params,)


@app.cell(hide_code=True)
def _(mo):
    data_params = mo.ui.array([
        mo.ui.number(start=1, step=1, label="Dataset ID"),
    ])
    data_params
    return (data_params,)


@app.cell(hide_code=True)
def _(mo):
    conn_params = mo.ui.array([
        mo.ui.text(label="OMERO username"),
        mo.ui.text(label="OMERO password", kind="password"),
        mo.ui.text(value="omero.mri.cnrs.fr", label="OMERO server URL"),
        mo.ui.number(value=4064, start=1, step=1, label="OMERO port"),
        mo.ui.text(value="CND project", label="Group"),
        mo.ui.checkbox(value=True, label="Secured connection")
    ])
    conn_params
    return (conn_params,)


@app.cell
def _(
    analysis_functions,
    analysis_params,
    conn_params,
    data_params,
    np,
    omero_tb,
):
    conn = omero_tb.open_connection(
        username=conn_params[0].value,
        password=conn_params[1].value,   
        host=conn_params[2].value,
        port=int(conn_params[3].value),
        group=conn_params[4].value,
        secure=conn_params[5].value
    )
    print("Connected") if conn.connect() else print("Unable to connect")

    try:
        dataset = omero_tb.get_dataset(connection=conn, dataset_id=data_params[0].value)
        dataset_images = omero_tb.get_dataset_images(dataset=dataset)
    
        for image in dataset_images:
            image_name = image.getName()
            if not image_name.endswith("SIR.dv"):
                continue
            print(f"Analyzing: {image_name}")
            image_data = omero_tb.get_intensities(image, c_range=int(analysis_params[2].value))
            image_data = analysis_functions.rescale_SIM(image_data)
            image_data = np.squeeze(image_data)
            spots = analysis_functions.find_spots(
                image_data, 
                min_sigma=analysis_params[0].value, 
                max_sigma=analysis_params[1].value
            )
            points_with_zc = []
            points_preview = []
            for spot in spots:
                points_with_zc.append(
                    omero_tb.create_shape_point(x_pos=spot[2], y_pos=spot[1], z_pos=spot[0], c_pos=int(analysis_params[2].value))
                )
                points_preview.append(
                    omero_tb.create_shape_point(x_pos=spot[2], y_pos=spot[1])
                )
    
            omero_tb.create_roi(conn, image, points_with_zc)
            print(f"Found {len(points_with_zc)} spots. Creating preview image")
            image_data_mip = np.max(image_data, 0)
            image_mip = omero_tb.create_image_from_numpy_array(
                connection=conn,
                data=image_data_mip,
                image_name=image_name[:-3] + "_preview_mip",
                dataset=dataset,
                source_image_id=image.getId(),
                channels_list=[int(analysis_params[2].value)]
            )
            image_mip = omero_tb.get_image(conn, image_mip.getId())
            omero_tb.create_roi(conn, image_mip, points_preview)
    
    finally:
        print("DONE!")
        conn.close()
    return


if __name__ == "__main__":
    app.run()
