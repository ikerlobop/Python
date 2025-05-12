import colorama
from colorama import Fore, Style
import sys
from OpenSSL import crypto
import os

colorama.init(autoreset=True)

# Spinner
def spinner_gen():
    while True:
        for cursor in '|/-\\':
            yield cursor

spinner = spinner_gen()

def spin():
    sys.stdout.write(next(spinner))
    sys.stdout.flush()
    sys.stdout.write('\b')

# Ruta fija al archivo .p12
p12FilePath = "C:/Users/UserPaco/Downloads/*.p12"

# Validación de argumentos
if len(sys.argv) != 3 or sys.argv[1] != "--wordlist":
    print(Fore.RED + Style.BRIGHT + 'Uso incorrecto:\n')
    print(Fore.YELLOW + '  python p12Cracker.py --wordlist "ruta/al/wordlist.txt"')
    sys.exit(1)

wordlist = sys.argv[2]

if not os.path.isfile(wordlist):
    print(Fore.RED + f"\nError: No se encontró el archivo de wordlist: {wordlist}")
    sys.exit(1)

if not os.path.isfile(p12FilePath):
    print(Fore.RED + f"\nError: No se encontró el archivo .p12: {p12FilePath}")
    sys.exit(1)

# Iniciar fuerza bruta
iterations = 0

print('\n' + Fore.CYAN + 'Brute forcing...\n')

with open(wordlist, 'r', encoding='utf-8') as fp:
    for line in fp:
        spin()
        guess = line.strip()
        try:
            p12 = crypto.load_pkcs12(open(p12FilePath, 'rb').read(), guess.encode('utf-8'))
        except crypto.Error:
            p12 = None

        iterations += 1

        if p12:
            print(Fore.BLUE + '\n' + '*' * 64)
            print(f'{Fore.GREEN}¡Éxito!{Fore.RESET} Contraseña encontrada tras {Fore.YELLOW}{iterations}{Fore.RESET} intentos.\n')
            print(f'La contraseña es: {Style.BRIGHT + Fore.RED}{guess}\n')
            print(Fore.BLUE + '*' * 64 + '\n')

            # Guardar resultado
            with open("resultado_crackeo.txt", "w", encoding="utf-8") as res_file:
                res_file.write(f"Contraseña encontrada: {guess}\nIntentos: {iterations}\n")
            sys.exit(0)

print(Fore.RED + '\nNo se pudo descifrar el archivo. Prueba con otra wordlist.\n')
sys.exit(0)
#### python p12Cracker.py --wordlist "C:/Users/Ainhoa_1/Desktop/Nueva carpeta/wordlist.txt" crear archivo wordlist.txt en carpeta
