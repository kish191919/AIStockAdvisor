import SwiftUI

struct MainTabView: View {
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView()
                .tabItem {
                    Image(systemName: "house.fill")
                    Text("Home")
                }
                .tag(0)
            
            AnalysisView(stockSymbol: "", language: "en")
                .tabItem {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                    Text("Analysis")
                }
                .tag(1)
            
            MarketView()
                .tabItem {
                    Image(systemName: "chart.bar.fill")
                    Text("Market")
                }
                .tag(2)
            
            HistoryView()
                .tabItem {
                    Image(systemName: "clock.fill")
                    Text("History")
                }
                .tag(3)
            
            WatchListView()
                .tabItem {
                    Image(systemName: "star.fill")
                    Text("Watch List")
                }
                .tag(4)
        }
    }
}

struct MainTabView_Previews: PreviewProvider {
    static var previews: some View {
        MainTabView()
    }
}
