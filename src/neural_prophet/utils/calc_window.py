class WindowTime:
    def __init__(self, tempo_amostragem, tamanho_janela):
        self.tempo_amostragem = tempo_amostragem
        self.tamanho_janela = tamanho_janela
    def calcular_janela(tempo_amostragem, tamanho_janela):
        """Esta função recebe o tempo de amostragem em segundos e o tamanho da janela em segundos,
        e retorna a janela de tempo em dias, minutos e segundos."""
        total_segundos = tamanho_janela / frequencia_amostragem
        dias = total_segundos // (24 * 3600)
        total_segundos = total_segundos % (24 * 3600)
        horas = total_segundos // 3600
        total_segundos %= 3600
        minutos = total_segundos // 60
        total_segundos %= 60
        segundos = total_segundos
        return int(dias), int(horas), int(minutos), int(segundos)