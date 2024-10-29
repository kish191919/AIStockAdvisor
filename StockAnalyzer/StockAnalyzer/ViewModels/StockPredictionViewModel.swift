// ViewModels/StockPredictionViewModel.swift

import Foundation
import CoreData
import Combine

class StockPredictionViewModel: ObservableObject {
    private let context: NSManagedObjectContext
    private var timer: Timer?
    
    init(context: NSManagedObjectContext = PersistenceController.shared.container.viewContext) {
        self.context = context
        setupUpdateTimer()
    }
    
    func savePrediction(symbol: String, decision: String, currentPrice: Double,
                       expectedNextDayPrice: Double, reason: String) {
        let fetchRequest = NSFetchRequest<NSManagedObject>(entityName: "StockPrediction")
        
        let prediction = NSEntityDescription.insertNewObject(forEntityName: "StockPrediction", into: context)
        
        prediction.setValue(UUID(), forKey: "id")
        prediction.setValue(symbol, forKey: "symbol")
        prediction.setValue(Date(), forKey: "analysisDate")
        prediction.setValue(decision, forKey: "decision")
        prediction.setValue(currentPrice, forKey: "currentPrice")
        prediction.setValue(expectedNextDayPrice, forKey: "expectedNextDayPrice")
        prediction.setValue(reason, forKey: "reason")
        prediction.setValue(false, forKey: "isUpdate")
        prediction.setValue(0.0, forKey: "actualClosedPrice")
        prediction.setValue(0.0, forKey: "accuracy")
        
        save()
    }
    
    private func save() {
        do {
            try context.save()
        } catch {
            print("Error saving context: \(error)")
        }
    }
    
    private func setupUpdateTimer() {
        // 매일 장 마감 후 실행 (미국 주식 기준 EST 16:00)
        timer = Timer.scheduledTimer(withTimeInterval: 3600, repeats: true) { [weak self] _ in
            self?.updateClosingPrices()
        }
    }
    
    private func updateClosingPrices() {
        let fetchRequest = NSFetchRequest<NSManagedObject>(entityName: "StockPrediction")
        fetchRequest.predicate = NSPredicate(format: "isUpdate == %@", NSNumber(value: false))
        
        do {
            let predictions = try context.fetch(fetchRequest)
            for prediction in predictions {
                if let analysisDate = prediction.value(forKey: "analysisDate") as? Date,
                   Calendar.current.isDateInYesterday(analysisDate) {
                    updateActualPrice(for: prediction)
                }
            }
        } catch {
            print("Error fetching predictions: \(error)")
        }
    }
    
    private func updateActualPrice(for prediction: NSManagedObject) {
        guard let symbol = prediction.value(forKey: "symbol") as? String else { return }
        
        // API로 종가 데이터 가져오기
        let urlString = "https://query1.finance.yahoo.com/v8/finance/chart/\(symbol)?interval=1d&range=1d"
        guard let url = URL(string: urlString) else { return }
        
        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            if let error = error {
                print("Error fetching data: \(error)")
                return
            }
            
            guard let data = data else { return }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let chart = json["chart"] as? [String: Any],
                   let result = (chart["result"] as? [[String: Any]])?.first,
                   let indicators = result["indicators"] as? [String: Any],
                   let quote = (indicators["quote"] as? [[String: Any]])?.first,
                   let closes = quote["close"] as? [Double],
                   let closingPrice = closes.last {
                    
                    DispatchQueue.main.async {
                        // 결과 업데이트
                        prediction.setValue(closingPrice, forKey: "actualClosedPrice")
                        prediction.setValue(true, forKey: "isUpdate")
                        
                        // 정확도 계산
                        if let expectedPrice = prediction.value(forKey: "expectedNextDayPrice") as? Double {
                            let accuracy = (1 - abs(closingPrice - expectedPrice) / expectedPrice) * 100
                            prediction.setValue(accuracy, forKey: "accuracy")
                        }
                        
                        self?.save()
                    }
                }
            } catch {
                print("Error parsing JSON: \(error)")
            }
        }.resume()
    }
    
    deinit {
        timer?.invalidate()
    }
}
