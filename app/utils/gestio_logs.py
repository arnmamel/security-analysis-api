import os
from datetime import datetime

def manage_log_files(log_directory, max_files=10):
    # Assegurem que el directori de registres existeix
    os.makedirs(log_directory, exist_ok=True)
    # Llistem tots els fitxers .log al directori
    log_files = [file for file in os.listdir(log_directory) if file.endswith('.log')]
    # Ordenem els fitxers per data de creació (els més antics primer)
    log_files.sort(key=lambda f: os.path.getctime(os.path.join(log_directory, f)))
    # Elimina el fitxer de registre més antic si hi ha massa fitxers
    while len(log_files) >= max_files:
        os.remove(os.path.join(log_directory, log_files.pop(0)))
    # Creem un nou fitxer de registre amb una marca de temps única
    new_log_file = os.path.join(log_directory, f"logfile_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    return new_log_file