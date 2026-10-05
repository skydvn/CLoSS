import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)

    if args.prefix is not None:
        del param["prefix"]

    overrides = parse_overrides(args.set)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    args.update(overrides) # Command-line overrides, e.g. --set ssi_mode=global lambda_grow=1.0
    args.pop("set", None)

    train(args)

def parse_overrides(items):
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--prefix', '-p', type=str, default=None, help='prefix of config file.')        
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='Override config entries; values are parsed as JSON when possible.')
    return parser

if __name__ == '__main__':
    main()
