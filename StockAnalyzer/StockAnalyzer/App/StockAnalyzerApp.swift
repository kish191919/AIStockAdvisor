
//  App/StockAnalyzerApp.swift

import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

@main
struct StockAnalyzerApp: App {
    let persistenceController = PersistenceController.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
