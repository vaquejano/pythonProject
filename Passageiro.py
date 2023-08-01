class Passageiro:
    def __init__(self, nome, idade):
        self._nome = nome
        self._idade = idade

    def get_nome(self):
        return self._nome

    def set_nome(self, nome):
        self._nome = nome

    def get_idade(self):
        return self._idade

    def set_idade(self, idade):
        self._idade = idade

    def to_string(self):
        print(self._nome + ', \t' + str(self._idade))
