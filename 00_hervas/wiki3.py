import mwxml
import bz2

# Archivo comprimido de Wikipedia
dump_file = "C:/Users/usuario/Downloads/cawiki-latest-pages-articles.xml.bz2"
output_file = "wikipedia_ca_texto.txt"

# Abrir el archivo comprimido correctamente
with bz2.open(dump_file, "rb") as compressed_file, open(output_file, "w", encoding="utf-8") as out_file:
    dump = mwxml.Dump.from_file(compressed_file)
    
    for page in dump:
        # Solo artículos (namespace 0) y sin redirecciones
        if not page.redirect and page.namespace == 0:
            for revision in page:
                if revision.text:
                    # Escribir el texto del artículo al archivo
                    out_file.write(revision.text + "\n")
