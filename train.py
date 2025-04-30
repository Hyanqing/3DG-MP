from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch

from tools import from_scaler

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    for step, batch in enumerate(loader):
        # batch.x = torch.hstack((batch.x, batch.positions * 10000))

        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.positions).squeeze()
        y = batch.y.squeeze().float()
        reg_criterion = torch.nn.MSELoss()
        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval(model, device, loader, scaler):
    model.eval()
    y_true, y_pred, all_idx = [], [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.positions).squeeze(1)

        true = batch.y.view(pred.shape)
        idx = batch.id.view(pred.shape)
        y_true.append(true)
        y_pred.append(pred)
        all_idx.append(idx)
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    all_idx = torch.cat(all_idx, dim=0).cpu().numpy()
    r2 = r2_score(from_scaler(scaler, y_true), from_scaler(scaler, y_pred))
    rmse = mean_squared_error(from_scaler(scaler, y_true), from_scaler(scaler, y_pred), squared=False)
    mae = mean_absolute_error(from_scaler(scaler, y_true), from_scaler(scaler, y_pred))
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}, y_true, y_pred, all_idx


def pred(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            _pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.positions).squeeze(1)
        y_pred.append(_pred)

    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    return y_pred