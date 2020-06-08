import os
import click

from dataset.dataset import create_img_from_dir


@click.command()
@click.option("--records", prompt="list of records, sep by ','",
              help="list of records, sep by ','")
def create_imgs(records):
    if records == "all":
        create_img_from_dir(verbose=True)
    else:
        create_img_from_dir(records_wanted=records.split(","), verbose=True)


if __name__ == "__main__":
    create_imgs()
