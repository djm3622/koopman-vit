import torch
import torch.optim as optim
from tqdm.notebook import tqdm

def check_point(val_loss, best_val_loss, model, patience_counter, save_path):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        return patience_counter, val_loss
    else:
        patience_counter += 1
    return patience_counter, best_val_loss



def train(train, valid, model, epochs, patience, criterion, lr, save_path, step, train_log, valid_log, device, loading_bar=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-3)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        if loading_bar:
            loader = tqdm(train, desc=f'Training', leave=False, mininterval=2.0)
        else:
            loader = train
        
        for train_batch in loader:
            loss = step(train_batch, model, criterion, device)

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if loading_bar:
                loader.set_postfix(train_loss=loss.item())
                            
        train_loss /= len(train)
        train_log.append(train_loss)

        model.eval()
        val_loss = 0
        if loading_bar:
            loader = tqdm(valid, desc=f'Validation', leave=False, mininterval=2.0)
        else:
            loader = valid
        
        with torch.no_grad():
            for valid_batch in loader:
                loss = step(valid_batch, model, criterion, device)

                val_loss += loss.item()
                
                if loading_bar:
                    loader.set_postfix(val_loss=loss.item())
                                    
        val_loss /= len(valid)
        valid_log.append(val_loss)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        patience_counter, best_val_loss = check_point(val_loss, best_val_loss, model, patience_counter, save_path)

        if patience_counter >= patience:
            print('Early stopping triggered')
            break
            
            
def check_point_accelerate(val_loss, best_val_loss, model, patience_counter, save_path, accelerator):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        accelerator.save_model(model, save_path)
        return patience_counter, val_loss
    else:
        patience_counter += 1
    return patience_counter, best_val_loss


def accelerator_train(accelerator, train, valid, model, epochs, patience, criterion, save_path, step, 
                      train_log, valid_log, optimizer, scheduler, loading_bar=False):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        if loading_bar:
            loader = tqdm(train, desc=f'Training', leave=False, mininterval=2.0)
        else:
            loader = train
        
        for train_batch in loader:
            with accelerator.accumulate(model):
                loss = step(train_batch, model, criterion)

                optimizer.zero_grad() 
                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.item()
            
            if loading_bar:
                loader.set_postfix(train_loss=loss.item())
                            
        train_loss /= len(train)
        train_log.append(train_loss)

        model.eval()
        val_loss = 0
        if loading_bar:
            loader = tqdm(valid, desc=f'Validation', leave=False, mininterval=2.0)
        else:
            loader = valid
        
        with torch.no_grad():
            for valid_batch in loader:
                loss = step(valid_batch, model, criterion)

                val_loss += loss.item()
                
                if loading_bar:
                    loader.set_postfix(val_loss=loss.item())
                                    
        val_loss /= len(valid)
        valid_log.append(val_loss)
        scheduler.step(val_loss)

        
        accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        patience_counter, best_val_loss = check_point_accelerate(val_loss, best_val_loss, model, patience_counter, save_path, accelerator)

        if patience_counter >= patience:
            print('Early stopping triggered')
            break
