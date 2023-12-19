METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a machine translation evaluation metric, which is calculated based on the harmonic mean of precision and recall, with recall weighted more than precision.

## Inputs

METEOR has two mandatory arguments:

    Model Generated: daftar teks untuk dinilai, setiap teks harus berupa string dengan token yang dipisahkan spasi.
    Target: daftar target, dalam kasus (satu target per hasil model), atau daftar target (dalam kasus beberapa target per hasil model). Setiap target harus berupa string dengan token dipisahkan oleh spasi.

It also has several optional parameters:

    alpha: Parameter untuk mengendalikan bobot relatif dari presisi dan recall. Default: 0.9.
    beta: Parameter untuk mengendalikan bentuk penalti sebagai fungsi dari fragmentasi. Default: 3.
    gamma: Bobot relatif yang diberikan untuk penalti fragmentasi. Default: 0.5.
