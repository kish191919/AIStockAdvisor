import Foundation

@MainActor
class StockSymbolValidator: ObservableObject {
    @Published var isValid = true
    @Published var errorMessage = ""
    
    // 기본적인 심볼 포맷 검증
    func validateSymbolFormat(_ symbol: String) -> Bool {
        // 빈 문자열 체크
        guard !symbol.isEmpty else {
            errorMessage = "Please enter a stock symbol"
            isValid = false
            return false
        }
        
        // 길이 체크 (일반적으로 1-5자)
        guard (1...5).contains(symbol.count) else {
            errorMessage = "Stock symbol should be 1-5 characters"
            isValid = false
            return false
        }
        
        // 허용되는 문자만 포함하는지 체크 (알파벳, 숫자, 일부 특수문자)
        let allowedCharacterSet = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        guard symbol.uppercased().rangeOfCharacter(from: allowedCharacterSet.inverted) == nil else {
            errorMessage = "Invalid characters in stock symbol"
            isValid = false
            return false
        }
        
        isValid = true
        errorMessage = ""
        return true
    }
    
    // 야후 파이낸스 API를 통한 심볼 존재 여부 확인
    func verifySymbolExists(_ symbol: String) async -> Bool {
        guard let url = URL(string: "https://query1.finance.yahoo.com/v8/finance/chart/\(symbol)?interval=1d&range=1d") else {
            await MainActor.run {
                errorMessage = "Invalid URL"
                isValid = false
            }
            return false
        }
        
        do {
            let (data, _) = try await URLSession.shared.data(for: URLRequest(url: url))
            let decoder = JSONDecoder()
            let response = try decoder.decode(YahooChartResponse.self, from: data)
            
            if response.chart.error != nil {
                await MainActor.run {
                    errorMessage = "Invalid stock symbol"
                    isValid = false
                }
                return false
            }
            
            await MainActor.run {
                isValid = true
                errorMessage = ""
            }
            return true
        } catch {
            await MainActor.run {
                errorMessage = "Failed to verify stock symbol"
                isValid = false
            }
            return false
        }
    }
}
