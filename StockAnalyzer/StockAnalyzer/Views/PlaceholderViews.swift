import SwiftUI

struct MarketView: View {
    var body: some View {
        NavigationView {
            VStack {
                Text("Market View Coming Soon")
                    .font(.title)
                    .foregroundColor(.secondary)
            }
            .navigationTitle("Market")
        }
    }
}

struct WatchListView: View {
    var body: some View {
        NavigationView {
            VStack {
                Text("Watch List Coming Soon")
                    .font(.title)
                    .foregroundColor(.secondary)
            }
            .navigationTitle("Watch List")
        }
    }
}
