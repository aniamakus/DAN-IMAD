import sys
import pandas as pd

def main(filename):
    print(filename)
    df = pd.read_csv(filename, names=["Konfiguracja", "Metryka", "2", "3", "4", "5", "6", "7", "8", "9"])
    print(df.to_latex(index=False))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Provide filename")
        sys.exit(1)

    main(sys.argv[1])

