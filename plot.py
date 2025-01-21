import matplotlib.pyplot as plt

# Ścieżka do pliku CSV
file_path = "output_ENV_1_more_hunger_ceil.csv"

# Wczytanie danych bez użycia Pandas
epoki = []
drapieznicy = []
ofiary = []

with open(file_path, 'r') as file:
    for line in file:
        values = line.strip().split(',')
        if len(values) == 3:
            epoka, drapieznik, ofiara = map(int, values)
            epoki.append(epoka)
            drapieznicy.append(drapieznik)
            ofiary.append(ofiara)

# Tworzenie wykresu
plt.figure(figsize=(8, 6))
plt.plot(epoki, drapieznicy, marker='o', label="Drapieżnicy")
plt.plot(epoki, ofiary, marker='s', label="Ofiary")

# Oznaczenia
plt.xlabel("Epoka")
plt.ylabel("Liczba")
plt.title("Populacja drapieżników i ofiar w czasie")
plt.legend()
plt.grid()

# Zapis do pliku
output_path = "populacja_z_pliku.jpg"
plt.savefig(output_path, dpi=300)

