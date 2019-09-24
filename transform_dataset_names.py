import argparse
from facematch.age_prediction.utils.metadata_transformer import MetadataTransformer

parser = argparse.ArgumentParser()


def prepare_image_batch():
    parser.add_argument("-d", "--sample_dir", help="Path to images folder")
    parser.add_argument("-o", "--output_directory", help="Path to save results")
    args = vars(parser.parse_args())

    names_transformer = MetadataTransformer(args)
    names_transformer.process_samples_directory()


if __name__ == "__main__":
    prepare_image_batch()
