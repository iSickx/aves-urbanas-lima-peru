"""
01_descargar_dataset.py
Descarga fotos de aves urbanas de Lima desde iNaturalist (observaciones de
"grado de investigación", es decir, identificación validada por la comunidad).

Primero busca observaciones dentro de Lima; si faltan fotos para una especie,
completa con observaciones de todo el Perú. Solo guarda fotos con licencia
Creative Commons y registra autor y licencia en metadatos.csv (para citar
la fuente de los datos en el informe).

Es REANUDABLE: si se corta, vuelve a ejecutarlo y continúa donde quedó
(omite las especies completas y no repite fotos ya descargadas).

Uso:
    pip install requests pillow
    python 01_descargar_dataset.py --por-especie 300 --salida data
    # solo algunas especies:
    python 01_descargar_dataset.py --solo "Falco sparverius,Larus dominicanus"
"""
import argparse
import csv
import io
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

API = "https://api.inaturalist.org/v1"

# (nombre científico, nombre común en Lima). Edita esta lista si tu grupo
# decide cambiar especies; el resto del pipeline se adapta solo.
ESPECIES = [
    ("Columba livia", "Paloma doméstica"),
    ("Columbina cruziana", "Tortolita peruana"),
    ("Zenaida meloda", "Cuculí"),
    ("Zonotrichia capensis", "Gorrión de collar rufo (chingolo)"),
    ("Passer domesticus", "Gorrión común"),
    ("Mimus longicaudatus", "Calandria de cola larga"),
    ("Turdus chiguanco", "Zorzal chiguanco"),
    ("Pyrocephalus obscurus", "Mosquero bermellón (turtupilín)"),
    ("Amazilia amazilia", "Amazilia costeña (colibrí)"),
    ("Falco sparverius", "Cernícalo americano"),
    ("Coragyps atratus", "Gallinazo de cabeza negra"),
    ("Larus dominicanus", "Gaviota dominicana"),
]

# Cajas geográficas aproximadas (lat/lng)
BBOX = {
    "lima": dict(swlat=-12.55, swlng=-77.25, nelat=-11.55, nelng=-76.55),
    "peru": dict(swlat=-18.40, swlng=-81.40, nelat=-0.03, nelng=-68.60),
}
LICENCIAS_OK = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}
CAMPOS = ["especie", "nombre_comun", "archivo", "obs_id", "foto_id",
          "licencia", "autor", "region", "url"]


def get(url, params=None, reintentos=4):
    r = None
    for i in range(reintentos):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 429:  # demasiadas peticiones
            time.sleep(10 * (i + 1))
        else:
            time.sleep(2)
    raise RuntimeError(f"Fallo al consultar {url} ({r.status_code if r is not None else 'sin respuesta'})")


def resolver_taxon(nombre):
    data = get(f"{API}/taxa", {"q": nombre, "rank": "species", "per_page": 5}).json()
    for t in data["results"]:
        if t["name"].lower() == nombre.lower():
            return t["id"]
    if data["results"]:
        t = data["results"][0]
        print(f"  AVISO: '{nombre}' no coincide exacto; se usa '{t['name']}'. Revisa.")
        return t["id"]
    return None


def buscar_fotos(taxon_id, region, cantidad, vistos):
    fotos, pagina = [], 1
    while len(fotos) < cantidad and pagina <= 20:
        params = dict(taxon_id=taxon_id, quality_grade="research", photos="true",
                      per_page=100, page=pagina, order_by="votes", **BBOX[region])
        resultados = get(f"{API}/observations", params).json()["results"]
        if not resultados:
            break
        for obs in resultados:
            if not obs.get("photos"):
                continue
            p = obs["photos"][0]  # una foto por observación evita casi-duplicados
            if p["id"] in vistos or p.get("license_code") not in LICENCIAS_OK:
                continue
            vistos.add(p["id"])
            fotos.append(dict(obs_id=obs["id"], foto_id=p["id"],
                              url=p["url"].replace("square", "medium"),
                              licencia=p["license_code"],
                              autor=p.get("attribution", ""), region=region))
            if len(fotos) >= cantidad:
                break
        pagina += 1
        time.sleep(1)  # iNaturalist pide máx. ~1 petición por segundo
    return fotos


def guardar(foto, destino):
    """Descarga una foto (si no existe ya). Devuelve True si quedó guardada."""
    if destino.exists():
        return True
    try:
        r = get(foto["url"])
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.save(destino, "JPEG", quality=92)
        return True
    except Exception as e:  # imagen corrupta o caída de red
        print(f"  omitida {foto['foto_id']}: {e}")
        return False


def leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-especie", type=int, default=300)
    ap.add_argument("--salida", default="data")
    ap.add_argument("--solo", default="",
                    help="nombres científicos separados por coma; vacío = todas")
    args = ap.parse_args()

    solo = {s.strip().lower() for s in args.solo.split(",") if s.strip()}
    lista = [e for e in ESPECIES if not solo or e[0].lower() in solo]
    salida = pathlib.Path(args.salida)
    salida.mkdir(exist_ok=True)

    for cientifico, comun in lista:
        carpeta = salida / cientifico.replace(" ", "_")
        carpeta.mkdir(exist_ok=True)
        meta = carpeta / "metadatos.csv"
        existentes = len(list(carpeta.glob("*.jpg")))
        print(f"\n{comun} ({cientifico})")
        if meta.exists() and existentes >= int(0.9 * args.por_especie):
            print(f"  ya completa ({existentes} fotos), se omite")
            continue

        tid = resolver_taxon(cientifico)
        if tid is None:
            print("  No se encontró el taxón, se omite.")
            continue
        vistos = set()
        fotos = buscar_fotos(tid, "lima", args.por_especie, vistos)
        n_lima = len(fotos)
        if len(fotos) < args.por_especie:
            fotos += buscar_fotos(tid, "peru", args.por_especie - len(fotos), vistos)
        print(f"  {len(fotos)} fotos ({n_lima} de Lima, {len(fotos) - n_lima} del resto de Perú)")

        with ThreadPoolExecutor(max_workers=8) as ex:
            oks = list(ex.map(lambda f: guardar(f, carpeta / f"{f['foto_id']}.jpg"), fotos))
        filas = [dict(especie=cientifico, nombre_comun=comun,
                      archivo=f"{f['foto_id']}.jpg", **f)
                 for f, ok in zip(fotos, oks) if ok]
        escribir_csv(meta, filas)
        print(f"  guardadas: {len(filas)}")

    # Une los metadatos de todas las especies en un solo archivo
    todas = []
    for m in sorted(salida.glob("*/metadatos.csv")):
        todas += leer_csv(m)
    escribir_csv(salida / "metadatos.csv", todas)
    print(f"\nListo: {len(todas)} imágenes en total. Metadatos en {salida / 'metadatos.csv'}")


if __name__ == "__main__":
    main()
