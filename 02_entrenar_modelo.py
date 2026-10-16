"""
02_entrenar_modelo.py
Transfer learning con MobileNetV2 (ImageNet) para clasificar aves urbanas de Lima.
Pensado para Google Colab (activar GPU: Entorno de ejecución > Cambiar tipo > GPU).

Colab:
    !pip install -q tf_keras tensorflowjs scikit-learn matplotlib
    !python 02_entrenar_modelo.py --datos data --salida salida

Genera en la carpeta de salida:
    modelo_aves_lima.h5, class_names.json, metricas.json,
    matriz_confusion.png, curvas_entrenamiento.png
Luego convierte a TensorFlow.js (ver README.md).
"""
import argparse
import json
import os
import pathlib
import random

# Keras 2 (tf_keras) para que la conversión a TensorFlow.js funcione sin problemas.
try:
    import tf_keras  # noqa: F401
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
layers = keras.layers
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

IMG, BATCH, SEED = 224, 32, 42
AUTOTUNE = tf.data.AUTOTUNE
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])


MIN_FOTOS = 20  # clases con menos fotos se excluyen (no alcanzan para entrenar/evaluar)


def listar(datos):
    raiz = pathlib.Path(datos)
    clases, items = [], []
    for c in sorted(p.name for p in raiz.iterdir() if p.is_dir()):
        fotos = sorted((raiz / c).glob("*.jpg"))
        print(f"{c}: {len(fotos)} fotos")
        if len(fotos) < MIN_FOTOS:
            print(f"  AVISO: '{c}' tiene muy pocas fotos; se excluye. Vuelve a ejecutar la descarga.")
            continue
        i = len(clases)
        clases.append(c)
        items += [(str(f), i) for f in fotos]
    return clases, items


def dividir(items, n_clases):
    """División estratificada 70% entrenamiento / 15% validación / 15% prueba."""
    rng = random.Random(SEED)
    tr, va, te = [], [], []
    for c in range(n_clases):
        fs = [x for x in items if x[1] == c]
        rng.shuffle(fs)
        a, b = int(0.70 * len(fs)), int(0.85 * len(fs))
        tr += fs[:a]
        va += fs[a:b]
        te += fs[b:]
    return tr, va, te


def cargar(ruta, etiqueta):
    img = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
    return tf.image.resize(img, (IMG, IMG)), etiqueta


def hacer_ds(items, entrenar):
    rutas = [p for p, _ in items]
    etiquetas = [l for _, l in items]
    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    if entrenar:
        ds = ds.shuffle(len(items), seed=SEED)
    ds = ds.map(cargar, num_parallel_calls=AUTOTUNE)
    if entrenar:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


def construir(n_clases):
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False,
                                          weights="imagenet")
    base.trainable = False
    entrada = keras.Input((IMG, IMG, 3))
    x = base(entrada, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(n_clases, activation="softmax")(x)
    return keras.Model(entrada, salida), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data")
    ap.add_argument("--salida", default="salida")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--epocas-ft", type=int, default=15)
    args = ap.parse_args()
    out = pathlib.Path(args.salida)
    out.mkdir(exist_ok=True)

    clases, items = listar(args.datos)
    tr, va, te = dividir(items, len(clases))
    print(f"{len(clases)} clases | train {len(tr)} | val {len(va)} | test {len(te)}")
    ds_tr, ds_va, ds_te = hacer_ds(tr, True), hacer_ds(va, False), hacer_ds(te, False)

    # Pesos por clase para compensar especies con menos fotos
    conteo = np.bincount([l for _, l in tr], minlength=len(clases))
    pesos = {i: len(tr) / (len(clases) * max(c, 1)) for i, c in enumerate(conteo)}

    modelo, base = construir(len(clases))
    compilar = lambda lr: modelo.compile(
        optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    # Fase 1: solo la cabeza clasificadora
    compilar(1e-3)
    h1 = modelo.fit(ds_tr, validation_data=ds_va, epochs=args.epocas, class_weight=pesos)

    # Fase 2: ajuste fino de las últimas 30 capas del backbone
    base.trainable = True
    for capa in base.layers[:-60]:
        capa.trainable = False
    compilar(3e-5)
    h2 = modelo.fit(ds_tr, validation_data=ds_va, epochs=args.epocas_ft, class_weight=pesos,
                    callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

    # Evaluación en el conjunto de prueba (nunca visto durante el entrenamiento)
    probs = modelo.predict(ds_te)
    y_true = np.array([l for _, l in te])
    y_pred = probs.argmax(1)
    top3 = np.mean([y_true[i] in np.argsort(probs[i])[-3:] for i in range(len(y_true))])
    acc = float(np.mean(y_true == y_pred))
    print(f"\nPrecisión test: {acc:.3f} | Top-3: {top3:.3f}")
    etiquetas = list(range(len(clases)))
    reporte = classification_report(y_true, y_pred, labels=etiquetas, target_names=clases,
                                    output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, labels=etiquetas, target_names=clases,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(clases)), clases, rotation=90)
    ax.set_yticks(range(len(clases)), clases)
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion.png", dpi=150)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for k, nombre in enumerate(["accuracy", "loss"]):
        axs[k].plot(h1.history[nombre] + h2.history[nombre], label="entrenamiento")
        axs[k].plot(h1.history["val_" + nombre] + h2.history["val_" + nombre], label="validación")
        axs[k].axvline(len(h1.history[nombre]) - 1, ls="--", c="gray")
        axs[k].set_title(nombre)
        axs[k].legend()
    fig.tight_layout()
    fig.savefig(out / "curvas_entrenamiento.png", dpi=150)

    modelo.save(out / "modelo_aves_lima.h5")
    (out / "class_names.json").write_text(json.dumps(clases, ensure_ascii=False, indent=2))
    (out / "metricas.json").write_text(json.dumps(
        dict(accuracy_test=acc, top3_test=float(top3), n_train=len(tr), n_val=len(va),
             n_test=len(te), por_clase=reporte), ensure_ascii=False, indent=2))
    print("\nModelo guardado. Convierte a TensorFlow.js con:")
    print(f"  tensorflowjs_converter --input_format=keras {out}/modelo_aves_lima.h5 web/model")
    print(f"  cp {out}/class_names.json web/model/")


if __name__ == "__main__":
    main()
