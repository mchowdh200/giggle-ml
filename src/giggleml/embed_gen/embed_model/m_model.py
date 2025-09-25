"""
M Model, deep sets architecture, hyenaDNA pre-processing

    (interval set -> sequence set --)-> sequence embeddings -> [M Model] -> (a single) set embedding

            | hyenaDNA  | M Model Core
            |           |   MLP, "phi"
            |           |     (!) activations NOT saved by default
            |           |
    ACGT... -> A-vec   --> B-vec    ---
    ACGT... -> A-vec   --> B-vec     | mean          MLP, "row"
    ACGT... -> A-vec   --> B-vec     |------> B-vec ------------> C-vec
    ACGT... -> A-vec   --> B-vec     |
    ACGT... -> A-vec   --> B-vec    ---
                   |
                   |
   could be memoized

"""
