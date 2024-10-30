import SwiftUI

struct SymbolAutocompleteView: View {
    let suggestions: [String]
    let onSelect: (String) -> Void
    
    var body: some View {
        if !suggestions.isEmpty {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(suggestions, id: \.self) { symbol in
                    Button(action: { onSelect(symbol) }) {
                        Text(symbol)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                    }
                    .foregroundColor(.primary)
                    
                    if symbol != suggestions.last {
                        Divider()
                    }
                }
            }
            .background(Color(.systemBackground))
            .cornerRadius(10)
            .shadow(radius: 2)
            .padding(.horizontal)
        }
    }
}

#Preview {
    SymbolAutocompleteView(
        suggestions: ["AAPL", "GOOGL", "MSFT"],
        onSelect: { _ in }
    )
}
