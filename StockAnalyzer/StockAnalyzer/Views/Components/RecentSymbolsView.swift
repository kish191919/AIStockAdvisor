import SwiftUI

struct RecentSymbolsView: View {
    @ObservedObject var manager: RecentSymbolsManager
    let onSymbolSelect: (String) -> Void
    
    var body: some View {
        if !manager.recentSymbols.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Recent Searches")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(manager.recentSymbols, id: \.self) { symbol in
                            Button(action: { onSymbolSelect(symbol) }) {
                                Text(symbol)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(Color.gray.opacity(0.2))
                                    .cornerRadius(15)
                            }
                            .foregroundColor(.primary)
                        }
                    }
                }
            }
            .padding(.horizontal)
        }
    }
}

#Preview {
    RecentSymbolsView(
        manager: RecentSymbolsManager(),
        onSymbolSelect: { _ in }
    )
}
