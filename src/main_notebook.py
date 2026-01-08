import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ### Import the necessary libraries
    """)
    return


@app.cell
def _():
    import marimo as mo
    import omero_toolbox as omero_tb
    from skimage.feature import blob_log
    return mo, omero_tb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Open a connection with the OMERO server
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    conn_params = mo.ui.array([
        mo.ui.text(label="OMERO username"),
        mo.ui.text(label="OMERO password", kind="password"),
        mo.ui.text(label="OMERO server URL"),
        mo.ui.number(value=4064, start=1, step=1, label="OMERO port"),
        mo.ui.text(label="Group"),
        mo.ui.checkbox(value=True, label="Secured connection")
    ])
    conn_params
    return (conn_params,)


@app.cell(hide_code=True)
def _(conn_params, omero_tb):
    conn = omero_tb.open_connection(
        username=conn_params[0].value,
        password=conn_params[1].value,   
        host=conn_params[2].value,
        port=int(conn_params[3].value),
        group=conn_params[4].value,
        secure=conn_params[5].value
    )
    print("Connected") if conn.connect() else print("Unable to connect")
    return (conn,)


@app.cell
def _(mo):
    data_params = mo.ui.array([
        mo.ui.number(start=1, step=1, label="Dataset ID"),
    ])
    data_params
    return (data_params,)


@app.cell
def _(conn, data_params, omero_tb):
    dataset = omero_tb.get_dataset(connection=conn, dataset_id=data_params[0].value)
    dataset_images = omero_tb.get_dataset_images(dataset=dataset)
    print([i.getName() for i in dataset_images])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
