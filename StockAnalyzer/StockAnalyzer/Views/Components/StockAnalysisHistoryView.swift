// Views/Components/StockAnalysisHistoryView.swift
//import SwiftUI
//import CoreData
//
//struct StockAnalysisHistoryView: View {
//    @Environment(\.managedObjectContext) private var viewContext
//    @FetchRequest(
//        entity: NSEntityDescription.entity(forEntityName: "StockPrediction", in: PersistenceController.shared.container.viewContext)!,
//        sortDescriptors: [NSSortDescriptor(key: "analysisDate", ascending: false)]
//    ) private var predictions: FetchedResults<NSManagedObject>
//    
//    // 주식 심볼별로 예측들을 그룹화
//    private var groupedPredictions: [String: [NSManagedObject]] {
//        Dictionary(grouping: predictions) { prediction in
//            prediction.value(forKey: "symbol") as? String ?? ""
//        }
//    }
//    
//    @State private var expandedSymbols: Set<String> = []
//    @State private var expandedPredictions: Set<UUID> = []
//    
//    var body: some View {
//        List {
//            ForEach(Array(groupedPredictions.keys.sorted()), id: \.self) { symbol in
//                DisclosureGroup(
//                    isExpanded: Binding(
//                        get: { expandedSymbols.contains(symbol) },
//                        set: { if $0 { expandedSymbols.insert(symbol) } else { expandedSymbols.remove(symbol) } }
//                    )
//                ) {
//                    // 각 예측에 대한 버튼들
//                    ForEach(groupedPredictions[symbol] ?? [], id: \.self) { prediction in
//                        VStack(spacing: 8) {
//                            Button(action: {
//                                if let id = prediction.value(forKey: "id") as? UUID {
//                                    if expandedPredictions.contains(id) {
//                                        expandedPredictions.remove(id)
//                                    } else {
//                                        expandedPredictions.insert(id)
//                                    }
//                                }
//                            }) {
//                                VStack(alignment: .leading, spacing: 4) {
//                                    if let date = prediction.value(forKey: "analysisDate") as? Date {
//                                        Text(dateFormatter.string(from: date))
//                                            .font(.subheadline)
//                                            .foregroundColor(.secondary)
//                                    }
//                                    
//                                    HStack {
//                                        Text("Decision: \(prediction.value(forKey: "decision") as? String ?? "")")
//                                        Spacer()
//                                        Text("$\(prediction.value(forKey: "currentPrice") as? Double ?? 0, specifier: "%.2f")")
//                                    }
//                                    
//                                    HStack {
//                                        Text("Expected: $\(prediction.value(forKey: "expectedNextDayPrice") as? Double ?? 0, specifier: "%.2f")")
//                                        Spacer()
//                                        if let isUpdate = prediction.value(forKey: "isUpdate") as? Bool, isUpdate {
//                                            Text("Actual: $\(prediction.value(forKey: "actualClosedPrice") as? Double ?? 0, specifier: "%.2f")")
//                                        } else {
//                                            Text("Actual: Pending")
//                                        }
//                                    }
//                                    
//                                    if let isUpdate = prediction.value(forKey: "isUpdate") as? Bool, isUpdate {
//                                        let accuracy = prediction.value(forKey: "accuracy") as? Double ?? 0
//                                        Text("Accuracy: \(accuracy, specifier: "%.1f")%")
//                                            .foregroundColor(
//                                                accuracy >= 90 ? .green :
//                                                accuracy >= 70 ? .orange : .red
//                                            )
//                                    }
//                                }
//                                .padding(8)
//                                .frame(maxWidth: .infinity, alignment: .leading)
//                                .background(Color.gray.opacity(0.1))
//                                .cornerRadius(8)
//                            }
//                            
//                            // Reason (확장/축소 가능)
//                            if let id = prediction.value(forKey: "id") as? UUID,
//                               expandedPredictions.contains(id) {
//                                Text(prediction.value(forKey: "reason") as? String ?? "")
//                                    .font(.body)
//                                    .padding(8)
//                                    .frame(maxWidth: .infinity, alignment: .leading)
//                                    .background(Color.blue.opacity(0.1))
//                                    .cornerRadius(8)
//                            }
//                        }
//                        .padding(.vertical, 4)
//                    }
//                } label: {
//                    HStack {
//                        Text(symbol)
//                            .font(.headline)
//                        
//                        Spacer()
//                        
//                        Text("\(groupedPredictions[symbol]?.count ?? 0) predictions")
//                            .font(.subheadline)
//                            .foregroundColor(.secondary)
//                    }
//                    .padding(.vertical, 4)
//                }
//            }
//        }
//        .navigationTitle("Prediction History")
//    }
//    
//    private let dateFormatter: DateFormatter = {
//        let formatter = DateFormatter()
//        formatter.dateStyle = .medium
//        formatter.timeStyle = .short
//        return formatter
//    }()
//}




//// Views/Components/StockAnalysisHistoryView.swift
//import SwiftUI
//import CoreData
//
//struct StockAnalysisHistoryView: View {
//    @Environment(\.managedObjectContext) private var viewContext
//    @FetchRequest(
//        entity: NSEntityDescription.entity(forEntityName: "StockPrediction", in: PersistenceController.shared.container.viewContext)!,
//        sortDescriptors: [NSSortDescriptor(key: "analysisDate", ascending: false)]
//    ) private var predictions: FetchedResults<NSManagedObject>
//    
//    var body: some View {
//        VStack {
//            if predictions.isEmpty {
//                Text("No predictions found")
//                    .foregroundColor(.secondary)
//                    .padding()
//            } else {
//                Text("Found \(predictions.count) predictions")
//                    .padding()
//                
//                List {
//                    ForEach(predictions, id: \.self) { prediction in
//                        VStack(alignment: .leading, spacing: 8) {
//                            Text("Symbol: \(prediction.value(forKey: "symbol") as? String ?? "Unknown")")
//                                .font(.headline)
//                            
//                            if let date = prediction.value(forKey: "analysisDate") as? Date {
//                                Text("Date: \(date)")
//                            }
//                            
//                            Text("Decision: \(prediction.value(forKey: "decision") as? String ?? "Unknown")")
//                            Text("Current Price: $\(prediction.value(forKey: "currentPrice") as? Double ?? 0, specifier: "%.2f")")
//                            Text("Expected: $\(prediction.value(forKey: "expectedNextDayPrice") as? Double ?? 0, specifier: "%.2f")")
//                            Text("Reason: \(prediction.value(forKey: "reason") as? String ?? "Unknown")")
//                        }
//                    }
//                }
//            }
//        }
//        .navigationTitle("Prediction History")
//        .onAppear {
//            print("Number of predictions: \(predictions.count)")
//            do {
//                let fetchRequest: NSFetchRequest<NSManagedObject> = NSFetchRequest(entityName: "StockPrediction")
//                let count = try viewContext.count(for: fetchRequest)
//                print("Total records in Core Data: \(count)")
//            } catch {
//                print("Failed to fetch count: \(error)")
//            }
//        }
//    }
//}



/// Views/Components/StockAnalysisHistoryView.swift
import SwiftUI
import CoreData

struct StockAnalysisHistoryView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @FetchRequest(
        entity: NSEntityDescription.entity(forEntityName: "StockPrediction", in: PersistenceController.shared.container.viewContext)!,
        sortDescriptors: [NSSortDescriptor(key: "analysisDate", ascending: false)]
    ) private var predictions: FetchedResults<NSManagedObject>
    
    @State private var expandedSymbols: Set<String> = []
    @State private var expandedPredictions: Set<UUID> = []
    
    var groupedPredictions: [String: [NSManagedObject]] {
        Dictionary(grouping: predictions) { prediction in
            prediction.value(forKey: "symbol") as? String ?? ""
        }
    }
    
    var body: some View {
        Group {
            if predictions.isEmpty {
                EmptyPredictionView()
            } else {
                PredictionListView(
                    groupedPredictions: groupedPredictions,
                    expandedSymbols: $expandedSymbols,
                    expandedPredictions: $expandedPredictions
                )
            }
        }
        .navigationTitle("Prediction History")
    }
}

// MARK: - Supporting Views
private struct EmptyPredictionView: View {
    var body: some View {
        VStack {
            Text("No predictions found")
                .foregroundColor(.secondary)
                .padding()
        }
    }
}

private struct PredictionListView: View {
    let groupedPredictions: [String: [NSManagedObject]]
    @Binding var expandedSymbols: Set<String>
    @Binding var expandedPredictions: Set<UUID>
    
    var body: some View {
        List {
            ForEach(Array(groupedPredictions.keys.sorted()), id: \.self) { symbol in
                PredictionGroupView(
                    symbol: symbol,
                    predictions: groupedPredictions[symbol] ?? [],
                    expandedSymbols: $expandedSymbols,
                    expandedPredictions: $expandedPredictions
                )
            }
        }
    }
}

private struct PredictionGroupView: View {
    let symbol: String
    let predictions: [NSManagedObject]
    @Binding var expandedSymbols: Set<String>
    @Binding var expandedPredictions: Set<UUID>
    
    var body: some View {
        DisclosureGroup(
            isExpanded: Binding(
                get: { expandedSymbols.contains(symbol) },
                set: { if $0 { expandedSymbols.insert(symbol) } else { expandedSymbols.remove(symbol) } }
            )
        ) {
            ForEach(predictions, id: \.self) { prediction in
                PredictionItemView(
                    prediction: prediction,
                    expandedPredictions: $expandedPredictions
                )
            }
        } label: {
            HStack {
                Text(symbol)
                    .font(.headline)
                Spacer()
                Text("\(predictions.count) predictions")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
}

private struct PredictionItemView: View {
    let prediction: NSManagedObject
    @Binding var expandedPredictions: Set<UUID>
    
    var body: some View {
        VStack {
            PredictionButton(
                prediction: prediction,
                expandedPredictions: $expandedPredictions
            )
            
            if let id = prediction.value(forKey: "id") as? UUID,
               expandedPredictions.contains(id) {
                PredictionReasonView(prediction: prediction)
            }
        }
    }
}

private struct PredictionButton: View {
    let prediction: NSManagedObject
    @Binding var expandedPredictions: Set<UUID>
    
    var body: some View {
        Button(action: {
            if let id = prediction.value(forKey: "id") as? UUID {
                if expandedPredictions.contains(id) {
                    expandedPredictions.remove(id)
                } else {
                    expandedPredictions.insert(id)
                }
            }
        }) {
            PredictionDetailView(prediction: prediction)
        }
    }
}

private struct PredictionDetailView: View {
    let prediction: NSManagedObject
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let date = prediction.value(forKey: "analysisDate") as? Date {
                Text(formatDate(date))
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            HStack {
                Text("Decision: \(prediction.value(forKey: "decision") as? String ?? "")")
                    .bold()
                Spacer()
                Text("$\(prediction.value(forKey: "currentPrice") as? Double ?? 0, specifier: "%.2f")")
            }
            
            Text("Expected: $\(prediction.value(forKey: "expectedNextDayPrice") as? Double ?? 0, specifier: "%.2f")")
            
            if let isUpdate = prediction.value(forKey: "isUpdate") as? Bool, isUpdate {
                PredictionAccuracyView(prediction: prediction)
            }
        }
        .padding(8)
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
    
    // 날짜 포맷팅 함수 추가
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

private struct PredictionAccuracyView: View {
    let prediction: NSManagedObject
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Actual: $\(prediction.value(forKey: "actualClosedPrice") as? Double ?? 0, specifier: "%.2f")")
            let accuracy = prediction.value(forKey: "accuracy") as? Double ?? 0
            Text("Accuracy: \(accuracy, specifier: "%.1f")%")
                .foregroundColor(
                    accuracy >= 90 ? .green :
                    accuracy >= 70 ? .orange : .red
                )
        }
    }
}

private struct PredictionReasonView: View {
    let prediction: NSManagedObject
    
    var body: some View {
        Text(prediction.value(forKey: "reason") as? String ?? "")
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.blue.opacity(0.1))
            .cornerRadius(8)
    }
}
