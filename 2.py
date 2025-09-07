import re

class Token:
    def __init__(self, tipo, valor):
        self.tipo = tipo
        self.valor = valor

    def __repr__(self):
        return f'Token({self.tipo}, {self.valor})'

def lexer(entrada):
    reglas = [
        ('NUMERO',       r'\d+'),
        ('SUMA',         r'\+'),
        ('RESTA',        r'-'),
        ('MULTIPLICACION', r'\*'),
        ('DIVISION',     r'/'),
        ('ASIGNACION',   r'='),
        ('ID',           r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('PARENT_IZQ',   r'\('),
        ('PARENT_DER',   r'\)'),
        ('ESPACIO',      r'\s+'),  # ignorar espacios
    ]
    
    tokens = []
    pos = 0
    while pos < len(entrada):
        match = None
        for tipo, patron in reglas:
            regex = re.compile(patron)
            match = regex.match(entrada, pos)
            if match:
                if tipo != 'ESPACIO':  # no guardamos los espacios
                    tokens.append(Token(tipo, match.group(0)))
                pos = match.end(0)
                break
        if not match:
            raise Exception(f"Carácter inesperado: {entrada[pos]}")
    return tokens


# Ejemplo de uso
entrada = "x = 10 + 5 * (2 - 3)"
tokens = lexer(entrada)
for token in tokens:
    print(token)
