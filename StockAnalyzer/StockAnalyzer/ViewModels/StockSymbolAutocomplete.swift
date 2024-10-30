import Foundation
import Combine

class StockSymbolAutocomplete: ObservableObject {
    @Published var suggestions: [String] = []
    private var task: Task<Void, Never>?
    
    func getSuggestions(for query: String) {
        guard query.count >= 1 else {
            suggestions = []
            return
        }
        
        // Cancel any existing task
        task?.cancel()
        
        task = Task {
            // Yahoo Finance API의 자동완성 엔드포인트 사용
            let urlString = "https://query1.finance.yahoo.com/v1/finance/search?q=\(query)&quotesCount=5&newsCount=0"
            guard let url = URL(string: urlString) else { return }
            
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                let decoded = try JSONDecoder().decode(SearchResponse.self, from: data)
                
                await MainActor.run {
                    self.suggestions = decoded.quotes.map { $0.symbol }
                }
            } catch {
                print("Autocomplete error: \(error)")
            }
        }
    }
}
