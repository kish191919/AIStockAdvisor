import Foundation

class RecentSymbolsManager: ObservableObject {
    @Published var recentSymbols: [String] = []
    private let maxSymbols = 5
    private let defaults = UserDefaults.standard
    private let key = "recentStockSymbols"
    
    init() {
        recentSymbols = defaults.stringArray(forKey: key) ?? []
    }
    
    func addSymbol(_ symbol: String) {
        var symbols = recentSymbols
        // 이미 있는 심볼이면 제거
        symbols.removeAll(where: { existingSymbol in
            existingSymbol == symbol
        })
        // 새 심볼 추가
        symbols.insert(symbol, at: 0)
        // 최대 개수 유지
        if symbols.count > maxSymbols {
            symbols = Array(symbols.prefix(maxSymbols))
        }
        recentSymbols = symbols
        defaults.set(symbols, forKey: key)
    }
}
