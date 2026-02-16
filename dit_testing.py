import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import dit
    from dit.xrdist import XRDistribution

    return (XRDistribution,)


@app.cell
def _(XRDistribution):
    d = XRDistribution(['000', '011', '101', '110'], [1/4]*4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
