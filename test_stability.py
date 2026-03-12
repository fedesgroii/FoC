import requests
import json

URL = "http://127.0.0.1:10000/api/stabilita"

tests = [
    {
        "name": "2x2 Asintoticamente Stabile (Continuo)",
        "matrix": [[-1, 0], [0, -2]],
        "dominio": "R",
        "expected": "Asintoticamente Stabile"
    },
    {
        "name": "2x2 Instabile (Continuo) - Radice Re > 0",
        "matrix": [[1, 0], [0, -2]],
        "dominio": "R",
        "expected": "Instabile"
    },
    {
        "name": "2x2 Stabile ma non asintoticamente (Continuo) - Radice Semplice Re = 0",
        "matrix": [[0, 1], [-1, 0]],  # Autovalori +/- i
        "dominio": "R",
        "expected": "Stabile (ma non asintoticamente)"
    },
    {
        "name": "2x2 Instabile (Continuo) - Radice Multipla Re = 0",
        "matrix": [[0, 1], [0, 0]],  # Autovalore 0 doppio
        "dominio": "R",
        "expected": "Instabile"
    },
    {
        "name": "2x2 Asintoticamente Stabile (Discreto) - Moduli < 1",
        "matrix": [[0.5, 0], [0, 0.8]],
        "dominio": "Z",
        "expected": "Asintoticamente Stabile"
    },
    {
        "name": "2x2 Instabile (Discreto) - Modulo > 1",
        "matrix": [[1.5, 0], [0, 0.8]],
        "dominio": "Z",
        "expected": "Instabile"
    },
    {
        "name": "2x2 Stabile ma non asintoticamente (Discreto) - Modulo Semplice = 1",
        "matrix": [[0, 1], [-1, 0]], # Autovalori +/- i (modulo 1)
        "dominio": "Z",
        "expected": "Stabile (ma non asintoticamente)"
    },
    {
        "name": "2x2 Instabile (Discreto) - Modulo Multiplo = 1",
        "matrix": [[1, 1], [0, 1]], # Autovalore 1 doppio
        "dominio": "Z",
        "expected": "Instabile"
    },
    {
        "name": "3x3 Asintoticamente Stabile (Continuo)",
        "matrix": [[-1, 0, 0], [0, -2, 0], [0, 0, -3]],
        "dominio": "R",
        "expected": "Asintoticamente Stabile"
    },
    {
        "name": "3x3 Stabile ma non asintoticamente (Continuo)",
        "matrix": [[-1, 0, 0], [0, 0, 1], [0, -1, 0]], # -1, i, -i
        "dominio": "R",
        "expected": "Stabile (ma non asintoticamente)"
    },
    {
        "name": "4x4 Instabile (Continuo) - Radici Multiple Immaginarie",
        "matrix": [[0, 1, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1, 0]], # +/- i doppi
        "dominio": "R",
        "expected": "Instabile"
    }
]

def run_tests():
    correct = 0
    print("Avvio dei Test per il backend di Stabilità\\n")
    for t in tests:
        payload = {
            "matrix": t["matrix"],
            "dominio": t["dominio"]
        }
        resp = requests.post(URL, json=payload)
        data = resp.json()
        
        status = data.get("stability_status", "")
        
        passed = t["expected"] in status
        
        print(f"Test: {t['name']}")
        print(f"Dominio: {t['dominio']}")
        print(f"Matrice: {t['matrix']}")
        print(f"Expected: {t['expected']}")
        print(f"Actual: {status}")
        
        if passed:
            print("✅ PASSED\\n")
            correct += 1
        else:
            print("❌ FAILED\\n")
            
    print(f"Risultato Finale: {correct}/{len(tests)} Tests Passati.")

if __name__ == '__main__':
    run_tests()
